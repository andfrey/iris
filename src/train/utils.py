# Evaluation function for regression models
from typing import Any, Dict, Tuple

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import wandb


def evaluate_regression(y, preds, prefix, wandb_run, plot=True):
    """
    Evaluate regression model on train and validation sets.
    Args:
        model: Trained regression model (must support predict)
        X_train, y_train: Training features and labels
        X_val, y_val: Validation features and labels
        fucci_scalers: Optional scaling array (applied to y and predictions)
        wandb_run: Optional W&B run for logging
        prefix: Metric prefix
    Returns:
        Dictionary of metrics
    """

    mse = mean_squared_error(y, preds)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)

    metrics = {
        f"{prefix}_mse": mse,
        f"{prefix}_mae": mae,
        f"{prefix}_r2": r2,
    }

    if wandb_run is not None:
        wandb_run.log(metrics)
    if plot and wandb_run is not None:
        log_plot(
            y,
            preds=preds,
            mode="error",
            log_scale=True,
            title="Validation Set Error Scatter Plot",
            wandb_run=wandb_run,
        )
        log_plot(
            y,
            preds=preds,
            mode="val_pred",
            log_scale=True,
            title=f"{prefix} Set: Intensity Predictions Scatter Plot",
            wandb_run=wandb_run,
        )
        log_plot(
            y,
            preds=preds,
            mode="true_pred",
            log_scale=False,
            title=f"{prefix} Set: Predicted vs True Scatter Plot",
            wandb_run=wandb_run,
        )
        log_plot(
            y,
            preds=preds,
            mode="error_label",
            log_scale=False,
            title=f"{prefix} Set: Error / Intensity Distribution",
            wandb_run=wandb_run,
        )
    return metrics


def log_plot(y, preds, mode="error", log_scale=True, title=None, wandb_run=None):
    """
    Unified scatter plot for error or validation predictions.
    Args:
        mode: "error" for error scatter, "val_pred" for validation prediction scatter
        log_scale: If True, use log-transformed values
        X: Features (default: validation set)
        y: True labels (default: validation set)
        title: Plot title (optional)
    """
    import matplotlib.pyplot as plt

    channels = [(0, "488"), (1, "561")]
    errors = y - preds
    errors_norm = np.linalg.norm(errors, axis=1)

    if mode == "error":
        plt.figure(figsize=(8, 6))
        if log_scale:
            sc = plt.scatter(
                np.log(y[:, 0]), np.log(y[:, 1]), c=errors_norm, cmap="Blues", alpha=0.7
            )
        else:
            sc = plt.scatter(y[:, 0], y[:, 1], c=errors_norm, cmap="Blues", alpha=0.7)
        plt.colorbar(sc, label="Norm Error")
        plt.xlabel(f"{'Log ' if log_scale else ''}488 Intensity")
        plt.ylabel(f"{'Log ' if log_scale else ''}561 Intensity")
        plt.title(title or "Validation Set Error Scatter Plot")
        if wandb_run is not None:
            wandb_run.log({"error_scatter": wandb.Image(plt)})
    elif mode == "val_pred":
        plt.figure(figsize=(8, 6))
        if log_scale:
            plt.scatter(np.log(preds[:, 0]), np.log(preds[:, 1]), alpha=0.7)
        else:
            plt.scatter(preds[:, 0], preds[:, 1], alpha=0.7)

        plt.xlabel(f"{'Log ' if log_scale else ''}488 Intensity")
        plt.ylabel(f"{'Log ' if log_scale else ''}561 Intensity")
        plt.title(title or "Validation Set Intensity Predictions Scatter Plot")
        if wandb_run is not None:
            wandb_run.log({"val_pred_scatter": wandb.Image(plt)})
    elif mode == "true_pred":
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        for ax, (i, label) in zip(axes, channels):
            x = np.log(y[:, i]) if log_scale else y[:, i]
            y_pred = np.log(preds[:, i]) if log_scale else preds[:, i]
            ax.scatter(x, y_pred, alpha=0.7, label=label)
            axis_min = min(x.min(), y_pred.min())
            axis_max = max(x.max(), y_pred.max())
            ax.plot(
                [axis_min, axis_max],
                [axis_min, axis_max],
                linestyle="--",
                color="red",
                linewidth=1,
            )
            ax.set_xlabel(f"{'Log ' if log_scale else ''}True Intensity ({label})")
            ax.set_ylabel(f"{'Log ' if log_scale else ''}Predicted Intensity ({label})")
            ax.set_title(f"{label}: Predicted vs True")

        plt.suptitle(title or "Validation Set: Predicted vs True Scatter Plot")
        if wandb_run is not None:
            wandb_run.log({"true_vs_pred_scatter": wandb.Image(plt)})
    elif mode == "error_label":
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for ax, (i, label) in zip(axes, channels):
            ax.scatter(y[:, i], errors[:, i], alpha=0.7, label=label)
            ax.set_xlabel(f"True Intensity ({label})")
            ax.set_ylabel(f"Error: True - Predicted ({label})")
            ax.set_title(f"{label}: Error / intensity distribution")

        plt.suptitle(title or "Validation Set: Error / Intensity Distribution")
        axes[2].scatter(np.linalg.norm(y, axis=1), errors_norm, alpha=0.7, label="Norm")
        axes[2].set_xlabel("True Intensity (Norm)")
        axes[2].set_ylabel("Error (Norm)")
        axes[2].set_title("Norm: Error / intensity distribution")
        if wandb_run is not None:
            wandb_run.log({"error_label_scatter": wandb.Image(plt)})
    else:
        raise ValueError(f"Invalid mode: {mode}")
    plt.tight_layout()
    plt.show()


def train_val_split(
    data: np.ndarray, labels: np.ndarray, split_ratio: float = 0.8, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data and labels into randomized training and validation sets.

    Args:
        data: Feature data array
        labels: Corresponding labels array
        split_ratio: Proportion of data to use for training
    Returns:
        X_train, y_train, X_val, y_val
    """
    # Shuffle indices
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
