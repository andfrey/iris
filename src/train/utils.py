# Evaluation function for regression models
from typing import Any, Dict, Tuple
import sys
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from sympy import im
import wandb

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.evaluation import fucci_color_comparison


def log_regression_plots(y, preds, wandb_run, prefix, is_cyclic=False):
    """
    Log a standard suite of regression plots to W&B.

    This automatically handles 1D targets (e.g., phase) and 2D targets (FUCCI intensities) and
    logs: true residual diagnostics, prediction scatter, true-vs-pred per target, and residual plots.

    Args:
        y: True target values
        preds: Predicted values
        wandb_run: Weights & Biases run for logging
        prefix: Prefix for plot names
        is_cyclic: If True, also create circular residual plot for phase data
    """
    # Filter out negative values in y and corresponding predictions
    mask = (y[:, 0] >= 0) & (y[:, 1] >= 0) if y.shape[1] == 2 else y[:, 0] >= 0
    y = y[mask]
    preds = preds[mask]

    log_plot(
        y,
        preds=preds,
        log_scale=False,
        title=f"{prefix} Set Diagnostics",
        wandb_run=wandb_run,
        prefix=prefix,
    )


def error_heatmap_plot(fig, ax, x, y_, errors_norm, error_label, bins=250, title=None):
    """
    Plot a 2D heatmap of mean errors over the value space.
    Args:
        x: 1D array for x-axis (e.g., true or predicted 488 intensity)
        y_: 1D array for y-axis (e.g., true or predicted 561 intensity)
        errors_norm: 1D array of error norms
        bins: Number of bins for the heatmap
        title: Optional plot title
    Returns:
        fig, ax: Matplotlib figure and axis
    """
    import matplotlib.pyplot as plt

    heatmap, xedges, yedges = np.histogram2d(x, y_, bins=bins, weights=errors_norm)
    counts, _, _ = np.histogram2d(x, y_, bins=bins)
    mean_error = np.divide(heatmap, counts, out=np.zeros_like(heatmap), where=counts > 0)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(mean_error.T, extent=extent, origin="lower", aspect="auto", cmap="hot")
    ax.set_xlabel("488 True Intensity")
    ax.set_ylabel("561 True Intensity")
    ax.set_title(title or "Mean Error Heatmap")
    fig.colorbar(im, ax=ax, label=error_label)
    plt.tight_layout()
    return fig, ax


def log_plot(y, preds, log_scale=True, title=None, wandb_run=None, prefix=None):
    """
    Create and log a suite of diagnostic plots for regression.

    Handles both 1D targets (e.g., phase) and 2D targets (intensities).
    Logs figures with keys:
      - {prefix}_true_residuals_plot
      - {prefix}_pred_scatter
      - {prefix}_true_vs_pred_scatter
      - {prefix}_residuals_plot
    """
    import matplotlib.pyplot as plt

    # Normalize shapes to (N, D)
    y = np.asarray(y)
    preds = np.asarray(preds)
    if y.ndim == 1:
        y = y[:, None]
    if preds.ndim == 1:
        preds = preds[:, None]

    n_targets = y.shape[1]
    if n_targets != 1:
        # Filter out negative values in y and corresponding predictions
        mask = (y[:, 0] >= 0) & (y[:, 1] >= 0)
        y = y[mask]
        preds = preds[mask]
    channels = [(i, lbl) for i, lbl in enumerate(["488", "561"][:n_targets])]
    errors = y - preds
    errors_norm = np.abs(errors[:, 0]) if n_targets == 1 else np.linalg.norm(errors, axis=1)

    # 1) True residuals diagnostics
    if n_targets != 1:
        x = y[:, 0]
        y_ = y[:, 1]
        fig, axes = plt.subplots(1, 6, figsize=(36, 6))
        for i, (e_vals, label) in enumerate(
            (
                (np.abs(errors[:, 0]), "Absolute residual 488 intensity"),
                (np.abs(errors[:, 1]), "Absolute residual 561 intensity"),
                (errors_norm, "Norm Residuals"),
            )
        ):
            i = i * 2
            sc = axes[i].scatter(x, y_, c=e_vals, cmap="Blues", alpha=0.7)
            axes[i].set_xlabel(f"488 True Intensity")
            axes[i].set_ylabel("561 True Intensity")
            axes[i].set_title(f"Residuals Scatter Plot {i+1}")
            fig.colorbar(sc, ax=axes[i], label=label)

            error_heatmap_plot(fig, axes[i + 1], x, y_, e_vals, error_label=label, bins=30)
        if wandb_run is not None:
            wandb_run.log({f"{prefix}_true_residuals_plot": wandb.Image(fig)})
        plt.close(fig)

    # 2) Prediction scatter field (and 1D true-vs-pred)
    if n_targets == 1:
        fig2 = plt.figure(figsize=(8, 6))
        y_true = y[:, 0]
        y_pred = preds[:, 0]
        abs_err = np.abs(errors[:, 0])
        sc = plt.scatter(y_true, y_pred, c=abs_err, cmap="Blues", alpha=0.7)
        axis_min = min(y_true.min(), y_pred.min())
        axis_max = max(y_true.max(), y_pred.max())
        plt.plot([axis_min, axis_max], [axis_min, axis_max], "r--", lw=1)
        plt.colorbar(sc, label="Absolute Residual")
        plt.xlabel("True Value")
        plt.ylabel("Predicted Value")
        plt.title("Predicted vs True (1D)")
        if wandb_run is not None:
            wandb_run.log({f"{prefix}_true_vs_pred_scatter": wandb.Image(fig2)})
        plt.close(fig2)
    else:
        fig2 = plt.figure(figsize=(8, 6))
        plt.scatter(y[:, 0], y[:, 1], c="gray", alpha=0.2, label="True Labels")
        sc = plt.scatter(
            preds[:, 0],
            preds[:, 1],
            c=errors_norm,
            cmap="Blues",
            alpha=0.7,
            label="Predictions",
        )
        plt.colorbar(sc, label="Norm Residuals")
        plt.xlabel("488 Predicted Intensity")
        plt.ylabel("561 Predicted Intensity")
        plt.title("Intensity Predictions Scatter Plot")
        plt.legend()
        if wandb_run is not None:
            wandb_run.log({f"{prefix}_pred_scatter": wandb.Image(fig2)})
        plt.close(fig2)

        # True vs Pred per-channel
        fig3, axes3 = plt.subplots(1, 2, figsize=(12, 6))
        for ax, (i, label) in zip(axes3, channels):
            xch = y[:, i]
            y_pred_ch = preds[:, i]
            ax.scatter(xch, y_pred_ch, alpha=0.7, label=label)
            axis_min = min(xch.min(), y_pred_ch.min())
            axis_max = max(xch.max(), y_pred_ch.max())
            ax.plot(
                [axis_min, axis_max], [axis_min, axis_max], linestyle="--", color="red", linewidth=1
            )
            ax.set_xlabel(f"True Intensity ({label})")
            ax.set_ylabel(f"Predicted Intensity ({label})")
            ax.set_title(f"{label}: Predicted vs True")
        plt.suptitle("Predicted vs True Scatter Plot")
        if wandb_run is not None:
            wandb_run.log({f"{prefix}_true_vs_pred_scatter": wandb.Image(fig3)})
        plt.close(fig3)

    # 3) Residuals plots
    if n_targets == 1:
        fig4, axes4 = plt.subplots(1, 2, figsize=(12, 5))
        y_pred = preds[:, 0]
        axes4[0].scatter(y_pred, errors[:, 0], alpha=0.7)
        axes4[0].set_xlabel("Predicted Value")
        axes4[0].set_ylabel("Residual: True - Predicted")
        axes4[0].set_title("Residuals vs Predicted (1D)")

        axes4[1].hist(errors[:, 0], bins=40, alpha=0.8, color="tab:gray")
        axes4[1].set_xlabel("Residual")
        axes4[1].set_ylabel("Count")
        axes4[1].set_title("Residuals Histogram")
        if wandb_run is not None:
            wandb_run.log({f"{prefix}_residuals_plot": wandb.Image(fig4)})
        plt.close(fig4)
    else:
        fig4, axes4 = plt.subplots(1, 3, figsize=(18, 6))
        for ax, (i, label) in zip(axes4, channels):
            ax.scatter(preds[:, i], errors[:, i], alpha=0.7, label=label)
            ax.set_xlabel(f"Predicted Intensity ({label})")
            ax.set_ylabel(f"Residuals: True - Predicted ({label})")
            ax.set_title(f"{label}: Residuals plot")
        axes4[2].scatter(np.linalg.norm(preds, axis=1), errors_norm, alpha=0.7, label="Norm")
        axes4[2].set_xlabel("Predicted Intensity (Norm)")
        axes4[2].set_ylabel("Residuals (Norm)")
        axes4[2].set_title("Norm: Residuals / intensity distribution")
        if wandb_run is not None:
            wandb_run.log({f"{prefix}_residuals_plot": wandb.Image(fig4)})
        plt.close(fig4)

    # 4) Ground Truth Residuals Plot (for 2D intensity data)
    if n_targets == 2:
        fig5, axes5 = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 1: Residuals in 488nm vs ground truth 488nm
        scatter1 = axes5[0].scatter(
            y[:, 0],
            errors[:, 0],
            alpha=0.6,
            s=30,
            c=np.abs(errors[:, 0]),
            cmap="viridis",
            edgecolors="k",
            linewidth=0.5,
        )
        axes5[0].axhline(y=0, color="r", linestyle="--", linewidth=2, alpha=0.7)
        axes5[0].set_xlabel("Ground Truth 488nm Intensity", fontsize=11)
        axes5[0].set_ylabel("Residual: True - Pred (488nm)", fontsize=11)
        axes5[0].set_title("488nm Channel Residuals vs Ground Truth", fontsize=12)
        axes5[0].grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=axes5[0], label="|Residual|")

        # Plot 2: Residuals in 561nm vs ground truth 561nm
        scatter2 = axes5[1].scatter(
            y[:, 1],
            errors[:, 1],
            alpha=0.6,
            s=30,
            c=np.abs(errors[:, 1]),
            cmap="viridis",
            edgecolors="k",
            linewidth=0.5,
        )
        axes5[1].axhline(y=0, color="r", linestyle="--", linewidth=2, alpha=0.7)
        axes5[1].set_xlabel("Ground Truth 561nm Intensity", fontsize=11)
        axes5[1].set_ylabel("Residual: True - Pred (561nm)", fontsize=11)
        axes5[1].set_title("561nm Channel Residuals vs Ground Truth", fontsize=12)
        axes5[1].grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=axes5[1], label="|Residual|")

        # Plot 3: Norm of residuals vs norm of ground truth
        gt_norm = np.linalg.norm(y, axis=1)
        scatter3 = axes5[2].scatter(
            gt_norm,
            errors_norm,
            alpha=0.6,
            s=30,
            c=errors_norm,
            cmap="plasma",
            edgecolors="k",
            linewidth=0.5,
        )
        axes5[2].axhline(y=0, color="r", linestyle="--", linewidth=2, alpha=0.7)
        axes5[2].set_xlabel("Ground Truth Intensity Norm", fontsize=11)
        axes5[2].set_ylabel("Residual Norm", fontsize=11)
        axes5[2].set_title("Combined Residual Norm vs Ground Truth", fontsize=12)
        axes5[2].grid(True, alpha=0.3)
        plt.colorbar(scatter3, ax=axes5[2], label="Residual Norm")

        # Add statistics
        mae_488 = np.mean(np.abs(errors[:, 0]))
        mae_561 = np.mean(np.abs(errors[:, 1]))
        mae_norm = np.mean(errors_norm)
        rmse_norm = np.sqrt(np.mean(errors_norm**2))

        stats_text = (
            f"488nm MAE: {mae_488:.4f}\n"
            f"561nm MAE: {mae_561:.4f}\n"
            f"Norm MAE: {mae_norm:.4f}\n"
            f"Norm RMSE: {rmse_norm:.4f}"
        )

        fig5.text(
            0.98,
            0.98,
            stats_text,
            transform=fig5.transFigure,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.suptitle("Residuals vs Ground Truth Intensity", fontsize=14, y=1.02)
        plt.tight_layout()

        if wandb_run is not None:
            wandb_run.log({f"{prefix}_gt_residuals_plot": wandb.Image(fig5)})
        plt.close(fig5)

        # 5) FUCCI Color Distribution Comparison
        fucci_color_comparison(y, preds, wandb_run=wandb_run, prefix=prefix)
