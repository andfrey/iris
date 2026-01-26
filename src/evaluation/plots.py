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
    title_fontsize: int = 22
    label_fontsize: int = 20
    tick_fontsize: int = 18
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
        n_bins: int = 25,
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
            if y_true.ndim > 1:
                y_true = y_true.flatten()
                y_pred = y_pred.flatten()

            y_pred_scaled = np.where((y_pred - y_true) > 0.5, y_pred - 1.0, y_pred)
            y_pred_scaled = np.where((y_pred - y_true) < -0.5, y_pred + 1.0, y_pred_scaled)

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 10), dpi=self.config.dpi)

            scatter = ax1.scatter(
                y_true,
                y_pred_scaled,
                alpha=self.config.alpha,
                s=20,
                c=np.abs(residuals),
                cmap="Blues",
            )
            ax1.plot(
                [y_true.min(), y_true.max()],
                [y_true.min(), y_true.max()],
                "r--",
                lw=2,
                label="Perfect prediction",
            )

            cbar = plt.colorbar(scatter, ax=ax1)
            cbar.set_label(
                "Geodesic Distance True vs. Predicted", fontsize=self.config.label_fontsize
            )
            cbar.ax.tick_params(labelsize=self.config.tick_fontsize)

            abs_y_pred_min_val = np.ceil(np.abs(np.min(y_pred_scaled)) / 0.1) * 0.1
            abs_y_pred_max_val = np.ceil((np.max(y_pred_scaled) - 1) / 0.1) * 0.1
            abs_y_pred_max_val = 0.0 if abs_y_pred_max_val <= 0.0 else abs_y_pred_max_val
            y_ticks_circ = np.concat(
                [
                    np.linspace(float(1.0 - abs_y_pred_min_val), 1.0, int(abs_y_pred_min_val * 10)),
                    np.linspace(0.0, 1.0, 11),
                    np.linspace(0.0, float(abs_y_pred_max_val), int(abs_y_pred_max_val * 10)),
                ]
            )
            y_ticks = np.concat(
                [
                    np.linspace(float(-abs_y_pred_min_val), 0.0, int(abs_y_pred_min_val * 10)),
                    np.linspace(0.0, 1.0, 11),
                    np.linspace(1.0, 1.0 + float(abs_y_pred_max_val), int(abs_y_pred_max_val * 10)),
                ]
            )
            ax1.set_xlim(0, 1)
            ax1.set_yticks(ticks=y_ticks, labels=[f"{x:.1f}" for x in y_ticks])
            ax1.set_aspect("equal", adjustable="box")
            ax1.set_xlabel("True Values", fontsize=self.config.label_fontsize)
            ax1.set_ylabel("Predicted Values", fontsize=self.config.label_fontsize)
            ax1.set_title(title, fontsize=self.config.title_fontsize)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis="both", labelsize=self.config.tick_fontsize)
            ax1.legend(fontsize=self.config.label_fontsize)

            # Bin the true phase values
            bins = np.linspace(0, 1, n_bins + 1)
            bin_centers = (bins[:-1] + bins[1:]) / 2

            mean_preds = []
            std_preds = []
            counts = []

            for i in range(n_bins):
                mask = (y_true >= bins[i]) & (y_true < bins[i + 1])
                if mask.sum() > 0:
                    # For circular data, handle wrapping around 0/1
                    preds_in_bin = y_pred[mask]
                    true_in_bin = y_true[mask]
                    wrapped_preds = np.where(
                        true_in_bin - preds_in_bin > 0.5, preds_in_bin + 1, preds_in_bin
                    )

                    mean_pred = wrapped_preds.mean()
                    std_pred = wrapped_preds.std()

                    mean_preds.append(mean_pred)
                    std_preds.append(std_pred)
                    counts.append(mask.sum())
                else:
                    mean_preds.append(np.nan)
                    std_preds.append(np.nan)
                    counts.append(0)

            mean_preds = np.array(mean_preds)
            std_preds = np.array(std_preds)
            counts = np.array(counts)

            # Filter out bins with no data
            valid_mask = ~np.isnan(mean_preds)
            bin_centers_valid = bin_centers[valid_mask]
            mean_preds_valid = mean_preds[valid_mask]
            std_preds_valid = std_preds[valid_mask]
            counts_valid = counts[valid_mask]

            # Plot mean predictions
            ax2.plot(
                bin_centers_valid,
                mean_preds_valid,
                "b-",
                linewidth=2,
                label="Mean Prediction",
                marker="o",
                markersize=4,
            )

            # Plot std bands
            ax2.fill_between(
                bin_centers_valid,
                mean_preds_valid - std_preds_valid,
                mean_preds_valid + std_preds_valid,
                alpha=0.3,
                color="steelblue",
                label="±1 Std Dev",
            )

            # Plot perfect prediction line
            ax2.plot([0, 1], [0, 1], "r--", linewidth=2, label="Perfect Prediction")

            # Add sample counts as text annotations (optional, for sparse regions)
            for i, (x, y, count) in enumerate(
                zip(bin_centers_valid, mean_preds_valid, counts_valid)
            ):
                if count < 10:  # Highlight bins with few samples
                    ax2.text(
                        x,
                        y,
                        f"n={count}",
                        fontsize=7,
                        ha="center",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5),
                    )

            ax2.set_xlabel("True Phase", fontsize=self.config.label_fontsize)
            ax2.set_ylabel("Predicted Phase", fontsize=self.config.label_fontsize)
            ax2.set_title(f"Mean Prediction vs True Phase", fontsize=self.config.title_fontsize)
            ax2.legend(loc="upper left")
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, 1)
            ax2.set_aspect("equal", adjustable="box")
            ax2.legend(loc="upper left", fontsize=self.config.label_fontsize)
            ax2.tick_params(axis="both", labelsize=self.config.tick_fontsize)

            # Add diagonal reference line
            ax2.plot([0, 1], [0, 1], "k:", linewidth=1, alpha=0.5)

            bins = np.linspace(0, 1, 11)
            residuals_by_bin = []
            bin_centers = []

            for i in range(10):
                mask = (y_true >= bins[i]) & (y_true < bins[i + 1])
                bin_centers.append((bins[i] + bins[i + 1]) / 2)
                residuals_by_bin.append(np.abs(np.array(residuals)[mask].squeeze()))

            bin_labels = [f"{bin_centers[i]:.2f}" for i in range(10)]

            # Create box plot
            bp = ax3.boxplot(
                residuals_by_bin,
                positions=bin_centers,
                widths=0.08,
                patch_artist=True,
                showfliers=True,
                flierprops=dict(marker="o", markersize=3, alpha=0.3, markerfacecolor="gray"),
            )

            # Color the boxes
            for patch in bp["boxes"]:
                patch.set_facecolor("steelblue")
                patch.set_alpha(0.7)

            # Add median line styling
            for median in bp["medians"]:
                median.set_color("darkred")
                median.set_linewidth(2)

            ax3.set_xlim(0, 1)
            ax3.set_xticks(bin_centers)
            ax3.set_xticklabels(bin_labels)
            ax3.set_aspect("equal", adjustable="box")
            ax3.set_xlabel("True Phase", fontsize=self.config.label_fontsize)
            ax3.set_ylabel("Absolute Geodesic Error", fontsize=self.config.label_fontsize)
            ax3.set_title("Error Distribution by Phase", fontsize=self.config.title_fontsize)
            ax3.grid(True, alpha=0.3, axis="y")
            ax3.tick_params(axis="both", labelsize=self.config.tick_fontsize)
            ax3.legend(fontsize=self.config.label_fontsize)

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

        Includes:
        - Residuals vs predictions
        - Residual distribution histogram
        - Mean prediction vs true phase with std bands

        Args:
            y_true: Ground truth phase values
            y_pred: Predicted phase values
            residuals: Geodesic residuals
            title: Plot title
            n_bins: Number of bins for aggregating true phase values
            **kwargs: Additional plot arguments

        Returns:
            Matplotlib figure
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        residuals = np.asarray(residuals).flatten()

        fig, axes = plt.subplots(1, 2, figsize=(20, 6), dpi=self.config.dpi)

        # Plot 1: Residuals vs predictions
        ax1 = axes[0]
        ax1.scatter(y_pred, residuals, alpha=self.config.alpha, s=20, c="steelblue")
        ax1.axhline(y=0, color="r", linestyle="--", lw=2, label="Zero residual")
        ax1.set_xlabel("Predicted Phase", fontsize=self.config.label_fontsize)
        ax1.set_ylabel("Geodesic Distance", fontsize=self.config.label_fontsize)
        ax1.set_title(f"{title} - vs Predictions", fontsize=self.config.title_fontsize)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Histogram of residuals
        ax2 = axes[1]
        ax2.hist(residuals, bins=50, alpha=0.7, edgecolor="black", color="steelblue")
        ax2.axvline(x=0, color="r", linestyle="--", lw=2, label="Zero residual")
        ax2.axvline(
            x=residuals.mean(),
            color="orange",
            linestyle="--",
            lw=2,
            label=f"Mean: {residuals.mean():.4f}",
        )
        ax2.set_xlabel("Geodesic Distance", fontsize=self.config.label_fontsize)
        ax2.set_ylabel("Frequency", fontsize=self.config.label_fontsize)
        ax2.set_title(f"{title} - Distribution", fontsize=self.config.title_fontsize)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def von_mises_kappa_uncertainty_plot(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        kappa_values: np.ndarray,
        title: str = "Von Mises Kappa Uncertainty",
        **kwargs,
    ) -> plt.Figure:
        """
        Create plot of Von Mises kappa uncertainty estimates.

        Args:
            y_true: Ground truth phase values
            kappa_values: Estimated kappa values from Von Mises model
            title: Plot title
            **kwargs: Additional plot arguments

        Returns:
            Matplotlib figure
        """
        y_true = np.asarray(y_true).flatten()
        kappa_values = np.asarray(kappa_values).flatten()
        y_pred = y_pred.flatten()

        fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        scatter = ax.scatter(
            y_true,
            y_pred,
            alpha=self.config.alpha,
            s=20,
            c=kappa_values,
            cmap="viridis",
        )
        ax.plot(
            [y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            "r--",
            lw=2,
            label="Perfect prediction",
        )
        plt.colorbar(scatter, label="Kappa Values (uncertainty)", ax=ax)
        ax.set_xlabel("True Values", fontsize=self.config.label_fontsize)
        ax.set_ylabel("Predicted Values", fontsize=self.config.label_fontsize)
        ax.set_title(title, fontsize=self.config.title_fontsize)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
