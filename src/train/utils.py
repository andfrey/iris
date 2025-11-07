# Evaluation function for regression models
from typing import Any, Dict, Tuple

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from sympy import im
import wandb


def evaluate_regression(y, preds, prefix, wandb_run, plot=True):
    """
    Compute regression metrics and optionally log diagnostic plots to W&B.

    Args:
        y (np.ndarray): True target values.
        preds (np.ndarray): Predicted values from the model.
        prefix (str): Prefix for metric names (e.g., 'val', 'test').
        wandb_run: Optional Weights & Biases run for logging metrics and plots.
        plot (bool): If True, log diagnostic plots to W&B.

    Returns:
        dict: Dictionary of regression metrics (MSE, MAE, R2) with prefix.
    """
    mse = mean_squared_error(y, preds)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)

    # y = np.where(y < 0.0001, 0.0001, y)
    # preds = np.where(preds < 0.0001, 0.0001, preds)
    # log_mse = mean_squared_error(np.log(y), np.log(preds))
    # log_mae = mean_absolute_error(np.log(y), np.log(preds))
    # log_r2 = r2_score(np.log(y), np.log(preds))

    metrics = {
        f"{prefix}_mse": mse,
        f"{prefix}_mae": mae,
        f"{prefix}_r2": r2,
        # f"{prefix}_log_mse": log_mse,
        # f"{prefix}_log_mae": log_mae,
        # f"{prefix}_log_r2": log_r2,
    }

    if wandb_run is not None:
        wandb_run.log(metrics)
    if plot and wandb_run is not None:
        log_regression_plots(y, preds, wandb_run, prefix)

    return metrics


def log_regression_plots(y, preds, wandb_run, prefix):
    """
    Log a standard suite of regression plots to W&B.

    This automatically handles 1D targets (e.g., phase) and 2D targets (FUCCI intensities) and
    logs: true residual diagnostics, prediction scatter, true-vs-pred per target, and residual plots.
    """
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
        log_scale: Whether axes are log-transformed
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
    channels = [(i, lbl) for i, lbl in enumerate(["488", "561"][:n_targets])]
    errors = y - preds
    errors_norm = np.abs(errors[:, 0]) if n_targets == 1 else np.linalg.norm(errors, axis=1)

    # 1) True residuals diagnostics
    if n_targets != 1:
        if log_scale:
            x = np.log(y[:, 0])
            y_ = np.log(y[:, 1])
        else:
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
            axes[i].set_xlabel(f"{'Log ' if log_scale else ''}488 True Intensity")
            axes[i].set_ylabel(f"{'Log ' if log_scale else ''}561 True Intensity")
            axes[i].set_title(f"Residuals Scatter Plot {i+1}")
            fig.colorbar(sc, ax=axes[i], label=label)

            error_heatmap_plot(fig, axes[i + 1], x, y_, e_vals, error_label=label, bins=30)
        if wandb_run is not None:
            wandb_run.log({f"{prefix}_true_residuals_plot": wandb.Image(fig)})
        plt.close(fig)

    # 2) Prediction scatter field (and 1D true-vs-pred)
    if n_targets == 1:
        fig2 = plt.figure(figsize=(8, 6))
        y_true = np.log(y[:, 0]) if log_scale else y[:, 0]
        y_pred = np.log(preds[:, 0]) if log_scale else preds[:, 0]
        abs_err = np.abs(errors[:, 0])
        sc = plt.scatter(y_true, y_pred, c=abs_err, cmap="Blues", alpha=0.7)
        axis_min = min(y_true.min(), y_pred.min())
        axis_max = max(y_true.max(), y_pred.max())
        plt.plot([axis_min, axis_max], [axis_min, axis_max], "r--", lw=1)
        plt.colorbar(sc, label="Absolute Residual")
        plt.xlabel(f"{'Log ' if log_scale else ''}True Value")
        plt.ylabel(f"{'Log ' if log_scale else ''}Predicted Value")
        plt.title("Predicted vs True (1D)")
        if wandb_run is not None:
            wandb_run.log({f"{prefix}_pred_scatter": wandb.Image(fig2)})
            wandb_run.log({f"{prefix}_true_vs_pred_scatter": wandb.Image(fig2)})
        plt.close(fig2)
    else:
        fig2 = plt.figure(figsize=(8, 6))
        if log_scale:
            plt.scatter(np.log(y[:, 0]), np.log(y[:, 1]), c="gray", alpha=0.2, label="True Labels")
            sc = plt.scatter(
                np.log(preds[:, 0]),
                np.log(preds[:, 1]),
                c=errors_norm,
                cmap="Blues",
                alpha=0.7,
                label="Predictions",
            )
        else:
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
        plt.xlabel(f"{'Log ' if log_scale else ''}488 Predicted Intensity")
        plt.ylabel(f"{'Log ' if log_scale else ''}561 Predicted Intensity")
        plt.title("Intensity Predictions Scatter Plot")
        plt.legend()
        if wandb_run is not None:
            wandb_run.log({f"{prefix}_pred_scatter": wandb.Image(fig2)})
        plt.close(fig2)

        # True vs Pred per-channel
        fig3, axes3 = plt.subplots(1, 2, figsize=(12, 6))
        for ax, (i, label) in zip(axes3, channels):
            xch = np.log(y[:, i]) if log_scale else y[:, i]
            y_pred_ch = np.log(preds[:, i]) if log_scale else preds[:, i]
            ax.scatter(xch, y_pred_ch, alpha=0.7, label=label)
            axis_min = min(xch.min(), y_pred_ch.min())
            axis_max = max(xch.max(), y_pred_ch.max())
            ax.plot(
                [axis_min, axis_max], [axis_min, axis_max], linestyle="--", color="red", linewidth=1
            )
            ax.set_xlabel(f"{'Log ' if log_scale else ''}True Intensity ({label})")
            ax.set_ylabel(f"{'Log ' if log_scale else ''}Predicted Intensity ({label})")
            ax.set_title(f"{label}: Predicted vs True")
        plt.suptitle("Predicted vs True Scatter Plot")
        if wandb_run is not None:
            wandb_run.log({f"{prefix}_true_vs_pred_scatter": wandb.Image(fig3)})
        plt.close(fig3)

    # 3) Residuals plots
    if n_targets == 1:
        fig4, axes4 = plt.subplots(1, 2, figsize=(12, 5))
        y_pred = np.log(preds[:, 0]) if log_scale else preds[:, 0]
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


def train_val_split(
    data: np.ndarray, labels: np.ndarray, split_ratio: float = 0.8, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Randomly split data and labels into training and validation sets.

    Args:
        data (np.ndarray): Feature data array.
        labels (np.ndarray): Corresponding labels array.
        split_ratio (float): Proportion of data to use for training (default 0.8).
        random_state (int): Seed for reproducibility.

    Returns:
        tuple: (X_train, y_train, X_val, y_val)
    """
    # Shuffle indices for random split
    np.random.seed(random_state)
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    train_size = int(len(data) * split_ratio)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]
    X_train = data[train_idx]
    y_train = labels[train_idx]
    X_val = data[val_idx]
    y_val = labels[val_idx]
    return X_train, y_train, X_val, y_val
