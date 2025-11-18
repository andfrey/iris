"""
Evaluation module for model checkpoints with support for cyclic phase predictions.

Provides comprehensive evaluation including:
- Standard regression metrics (R², MAE, MSE, RMSE)
- Cyclic metrics for phase predictions (circular MAE, MSE, R²)
- Rich visualizations including phase confusion matrices and polar plots
- Automatic conversion from 2D intensities to phases using FucciCurveProjector
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_pipeline.curve_projector import FucciCurveProjector
from src.train.utils import evaluate_regression, log_fucci_color_comparison
from src.evaluation.utils import (
    circular_distance,
    circular_mae,
    circular_mse,
    circular_r2,
    load_model_from_checkpoint,
    maybe_build_dataloader_from_ckpt,
)
from src.models.cnet import CNet


def evaluate_regression_cyclic(
    y_true_phases,
    y_pred_phases,
    y_true_intensities=None,
    y_pred_intensities=None,
    prefix: str = "test",
) -> Dict[str, float]:
    """
    Evaluate regression with support for cyclic (phase) data.

    Args:
        y_true: Ground truth values [N,] for phase or [N, 2] for intensities
        y_pred: Predicted values [N,] for phase or [N, 2] for intensities
        prefix: Metric name prefix
        is_phase: Whether data is phase (cyclic) or not

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    metrics = {}
    if y_true_intensities is not None and y_pred_intensities is not None:
        # Standard metrics
        metrics[f"{prefix}_mse"] = mean_squared_error(y_true_intensities, y_pred_intensities)
        metrics[f"{prefix}_mae"] = mean_absolute_error(y_true_intensities, y_pred_intensities)
        metrics[f"{prefix}_r2"] = r2_score(y_true_intensities, y_pred_intensities)
        metrics[f"{prefix}_rmse"] = np.sqrt(metrics[f"{prefix}_mse"])

    # Cyclic metrics for phase data
    metrics[f"{prefix}_circular_mae"] = circular_mae(y_true_phases, y_pred_phases)
    metrics[f"{prefix}_circular_mse"] = circular_mse(y_true_phases, y_pred_phases)
    metrics[f"{prefix}_circular_r2"] = circular_r2(y_true_phases, y_pred_phases)
    metrics[f"{prefix}_circular_rmse"] = np.sqrt(metrics[f"{prefix}_circular_mse"])

    return metrics


def plot_phase_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Phase Predictions",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Create scatter plot for phase predictions with circular error coloring.

    Args:
        y_true: True phases [N,] in radians
        y_pred: Predicted phases [N,] in radians
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Compute circular errors
    errors = circular_distance(y_true, y_pred)

    # Scatter plot
    scatter = ax.scatter(y_true, y_pred, c=errors, cmap="coolwarm", alpha=0.6, s=20)

    # Perfect prediction line
    ax.plot([-np.pi, np.pi], [-np.pi, np.pi], "k--", lw=2, alpha=0.5, label="Perfect prediction")

    # Formatting
    ax.set_xlabel("True Phase (radians)", fontsize=12)
    ax.set_ylabel("Predicted Phase (radians)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Circular Error (radians)", fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_phase_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 12,
    title: str = "Phase Confusion Matrix",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Create a polar heatmap showing how phases are confused.

    Args:
        y_true: True phases [N,] in radians
        y_pred: Predicted phases [N,] in radians
        n_bins: Number of angular bins
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    # Create bins
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Digitize
    true_bins = np.digitize(y_true, bins) - 1
    pred_bins = np.digitize(y_pred, bins) - 1

    # Clip to valid range
    true_bins = np.clip(true_bins, 0, n_bins - 1)
    pred_bins = np.clip(pred_bins, 0, n_bins - 1)

    # Create confusion matrix
    confusion = np.zeros((n_bins, n_bins))
    for t, p in zip(true_bins, pred_bins):
        confusion[t, p] += 1

    # Normalize by row (true phase)
    row_sums = confusion.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    confusion_norm = np.divide(confusion, row_sums, where=row_sums > 0)

    # Create polar plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="polar")

    # Create meshgrid for polar coordinates
    theta = np.linspace(-np.pi, np.pi, n_bins + 1)
    r = np.linspace(0, 1, n_bins + 1)
    Theta, R = np.meshgrid(theta, r)

    # Plot heatmap
    # Reshape confusion matrix for polar plotting
    im = ax.pcolormesh(Theta.T, R.T, confusion_norm, cmap="YlOrRd", shading="auto")

    # Highlight diagonal (correct predictions) with frames
    # In the transposed polar plot:
    # - Angular (theta) direction = true phase bins (rows in confusion matrix)
    # - Radial (r) direction = predicted phase bins (columns in confusion matrix)
    # Diagonal elements are at confusion[i, i], which means:
    # - theta corresponds to bin i (true phase)
    # - r corresponds to bin i (predicted phase)

    bin_width = 2 * np.pi / n_bins
    r_width = 1.0 / n_bins

    for i in range(n_bins):
        # Angular position for true phase bin i
        theta_start = bins[i]
        theta_end = bins[i + 1]

        # Radial position for predicted phase bin i
        # In the transposed plot (Theta.T, R.T), the radial bins go from inside to outside
        r_start = i * r_width
        r_end = (i + 1) * r_width

        # Draw frame around the diagonal cell
        theta_frame = [theta_start, theta_end, theta_end, theta_start, theta_start]
        r_frame = [r_start, r_start, r_end, r_end, r_start]
        ax.plot(theta_frame, r_frame, color="black", linewidth=2.5, zorder=10)

    # Formatting
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_ylim(0, 1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.1)
    cbar.set_label("Prediction Probability", fontsize=11)

    # Add radial labels
    ax.set_ylabel("Predicted Phase →", fontsize=10, labelpad=30)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_circular_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_samples: int = None,
    title: str = "Circular Phase Comparison",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Create polar plot showing true→predicted connections.

    Args:
        y_true: True phases [N,] in radians
        y_pred: Predicted phases [N,] in radians
        max_samples: Maximum number of samples to plot (for clarity)
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    # Subsample if needed
    if max_samples and len(y_true) > max_samples:
        indices = np.random.choice(len(y_true), max_samples, replace=False)
        y_true = y_true[indices]
        y_pred = y_pred[indices]

    # Compute errors for coloring
    errors = circular_distance(y_true, y_pred)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="polar")
    # Set theta limits to show full circle from -π to π
    ax.set_thetamin(-180)  # -π in degrees
    ax.set_thetamax(180)  # π in degrees

    # Set radians labels instead of angles
    ax.set_xticks(
        [
            0,
            np.pi / 4,
            np.pi / 2,
            3 * np.pi / 4,
            np.pi,
            -3 * np.pi / 4,
            -np.pi / 2,
            -np.pi / 4,
        ],
        [
            "0",
            "π/4",
            "π/2",
            "3π/4",
            "π",
            "-3π/4",
            "-π/2",
            "-π/4",
        ],
        fontsize=20,
    )

    ax.set_yticks([0.8, 1.0])
    # Draw connections from true to predicted
    for t, p, e in zip(y_true, y_pred, errors):
        color = plt.cm.coolwarm(e / np.pi)  # Color by error
        ax.plot([t, p], [0.8, 1.0], color=color, alpha=0.3, lw=0.5)

    # Plot true and predicted points
    ax.scatter(y_true, np.ones_like(y_true) * 0.8, c="green", s=20, alpha=0.6, label="True")
    ax.scatter(y_pred, np.ones_like(y_pred) * 1.0, c="blue", s=20, alpha=0.6, label="Predicted")

    # Formatting
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_ylim(0, 1.2)
    ax.set_title(title, fontsize=22, fontweight="bold", pad=20)
    ax.legend(loc="upper right", fontsize=20)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Circular Error Distribution",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot histogram of circular errors.

    Args:
        y_true: True phases [N,] in radians
        y_pred: Predicted phases [N,] in radians
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    errors = circular_distance(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(errors, bins=50, alpha=0.7, color="skyblue", edgecolor="black")
    axes[0].axvline(
        np.mean(errors),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(errors):.3f}",
    )
    axes[0].axvline(
        np.median(errors),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {np.median(errors):.3f}",
    )
    axes[0].set_xlabel("Circular Error (radians)", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].set_title("Error Distribution", fontsize=13)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Cumulative distribution
    sorted_errors = np.sort(errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    axes[1].plot(sorted_errors, cumulative, linewidth=2, color="navy")
    axes[1].set_xlabel("Circular Error (radians)", fontsize=12)
    axes[1].set_ylabel("Cumulative Probability", fontsize=12)
    axes[1].set_title("Cumulative Error Distribution", fontsize=13)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_intensity_to_phase_projection(
    y_true_intensities: np.ndarray,
    y_pred_intensities: np.ndarray,
    y_true_phases: np.ndarray,
    y_pred_phases: np.ndarray,
    projector: FucciCurveProjector,
    title: str = "Intensity to Phase Projection",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Visualize 2D intensity predictions and their phase projections.

    Args:
        y_true_intensities: True intensities [N, 2]
        y_pred_intensities: Predicted intensities [N, 2]
        y_true_phases: True phases [N,] in radians
        y_pred_phases: Predicted phases [N,] in radians
        projector: FucciCurveProjector instance
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Intensity scatter
    ax = axes[0]
    errors = circular_distance(y_true_phases, y_pred_phases)

    # True and predicted intensities
    ax.scatter(
        y_true_intensities[:, 0],
        y_true_intensities[:, 1],
        c="lightgray",
        s=30,
        alpha=0.4,
        label="True intensities",
    )
    if y_pred_intensities is not None:
        scatter = ax.scatter(
            y_pred_intensities[:, 0],
            y_pred_intensities[:, 1],
            c=errors,
            cmap="coolwarm",
            s=40,
            alpha=0.7,
            label="Predicted intensities",
        )
    else:
        scatter = ax.scatter(
            y_true_intensities[:, 0],
            y_true_intensities[:, 1],
            c=errors,
            cmap="coolwarm",
            s=40,
            alpha=0.7,
            label="True intensities",
        )

    # Plot centroid and reference circle
    if projector.centroid is not None:
        ax.plot(
            projector.centroid[0],
            projector.centroid[1],
            "k*",
            markersize=15,
            label="Centroid",
            zorder=5,
        )
        if projector.radius is not None:
            circle = plt.Circle(
                projector.centroid,
                projector.radius,
                fill=False,
                color="gray",
                linestyle="--",
                linewidth=2,
                label="Reference circle",
            )
            ax.add_patch(circle)

    ax.set_xlabel("488 Intensity", fontsize=12)
    ax.set_ylabel("561 Intensity", fontsize=12)
    ax.set_title("Intensity Space", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Phase Error (radians)", fontsize=10)

    # Right: Phase scatter
    ax = axes[1]
    scatter = ax.scatter(y_true_phases, y_pred_phases, c=errors, cmap="coolwarm", alpha=0.6, s=40)
    ax.plot([-np.pi, np.pi], [-np.pi, np.pi], "k--", lw=2, alpha=0.5, label="Perfect prediction")

    ax.set_xlabel("True Phase (radians)", fontsize=12)
    ax.set_ylabel("Predicted Phase (radians)", fontsize=12)
    ax.set_title("Phase Space", fontsize=13)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Phase Error (radians)", fontsize=10)

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def evaluate_checkpoint(
    ckpt_path: str,
    data_config_path: Optional[str] = None,
    split: str = "test",
    output_dir: str = "logs/checkpoint_eval",
    device: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate a model checkpoint on a dataset split with comprehensive metrics and visualizations.

    Args:
        ckpt_path: Path to model checkpoint
        data_config_path: Path to data config YAML (if None, tries to load from checkpoint)
        split: Data split to evaluate ('train', 'val', 'test')
        output_dir: Directory to save outputs
        device: Device for computation (auto-detected if None)
        max_samples: Maximum number of samples to evaluate (None for all)

    Returns:
        Dictionary with metrics and paths to saved visualizations
    """
    import yaml
    from src.data_pipeline.dataset import ModularCellDataModule

    output_dir = Path(output_dir) / Path(ckpt_path).stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading checkpoint from: {ckpt_path}")

    # Load model
    model = load_model_from_checkpoint(ckpt_path)
    model.eval().to(device)

    #
    phase_labels = True
    loader = maybe_build_dataloader_from_ckpt(ckpt_path, split)
    if loader is None:
        print("Failed to create dataloader from checkpoint.")
        return
    dataset = loader.dataset.dataset
    if dataset.projector is not None:
        phase_labels = True
        projector = dataset.projector
        dataset.projector = None
    else:
        projector = FucciCurveProjector(dataset=dataset)
        projector.fit_from_dataset(use_cache=True)

    print(f"Evaluating on {split} split...")

    # Collect predictions
    all_preds = []
    all_labels = []
    with torch.no_grad():
        # i = 0
        for batch in tqdm(loader, desc="Predicting"):
            # if i > 1:
            #     break
            # i += 1
            if isinstance(batch[0], (list, tuple)):
                inputs = tuple(x.to(device) for x in batch[0])
            else:
                inputs = (batch[0].to(device),)

            intensity_labels = batch[1].to(device)
            preds = model(*inputs)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(intensity_labels.cpu().numpy())

            if max_samples and sum(len(p) for p in all_preds) >= max_samples:
                break

    # Concatenate
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    if max_samples:
        y_pred = y_pred[:max_samples]
        y_true = y_true[:max_samples]

    print(f"Collected {len(y_pred)} predictions")
    print(f"Prediction shape: {y_pred.shape}, Label shape: {y_true.shape}")

    # Determine if we need phase projection
    is_2d_output = y_pred.shape[1] == 2 if y_pred.ndim > 1 else False

    results = {
        "split": split,
        "num_samples": len(y_pred),
        "output_shape": y_pred.shape,
        "label_shape": y_true.shape,
    }

    print("Creating FucciCurveProjector...")
    # Fit if not already fitted

    # Project predictions to phases if 2D
    if is_2d_output:
        y_pred_phases = np.array([projector.compute_phase(p) for p in y_pred])
        y_pred_intensities = y_pred.copy()
    else:
        y_pred_phases = y_pred.flatten()
        y_pred_intensities = None

    y_true_phases = np.array([projector.compute_phase(p) for p in y_true])
    y_true_intensities = y_true.copy()

    # Compute metrics on phases
    metrics = evaluate_regression_cyclic(
        y_true_phases,
        y_pred_phases,
        y_true_intensities,
        y_pred_intensities,
        prefix=split,
    )
    results["metrics"] = metrics

    print("\n" + "=" * 60)
    print(f"METRICS ({split.upper()} split)")
    print("=" * 60)
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
    print("=" * 60 + "\n")

    # Save metrics
    with open(output_dir / f"metrics_{split}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    mask_idx = (y_true_intensities[:, 0] > 0) & (y_true_intensities[:, 1] > 0)
    if y_pred_intensities is not None:
        y_pred_intensities = y_pred_intensities[mask_idx, :]
    y_true_intensities = y_true_intensities[mask_idx, :]
    y_true_phases = y_true_phases[mask_idx]
    y_pred_phases = y_pred_phases[mask_idx]

    # Generate visualizations
    print("Generating visualizations...")

    # 1. Phase scatter
    fig = plot_phase_scatter(
        y_true_phases,
        y_pred_phases,
        title=f"Phase Predictions",
        save_path=output_dir / f"phase_scatter_{split}.png",
    )
    plt.close(fig)

    # 2. Phase confusion matrix
    fig = plot_phase_confusion_matrix(
        y_true_phases,
        y_pred_phases,
        title=f"Phase Confusion Matrix",
        save_path=output_dir / f"phase_confusion_{split}.png",
    )
    plt.close(fig)

    # 3. Circular comparison
    fig = plot_circular_comparison(
        y_true_phases,
        y_pred_phases,
        title=f"Circular Phase Comparison",
        save_path=output_dir / f"circular_comparison_{split}.png",
    )
    plt.close(fig)

    # 4. Error distribution
    fig = plot_error_distribution(
        y_true_phases,
        y_pred_phases,
        title=f"Phase Error Distribution",
        save_path=output_dir / f"error_distribution_{split}.png",
    )
    plt.close(fig)

    # 5. Scatter plot of intensity with phase error
    fig = plot_intensity_to_phase_projection(
        y_true_intensities,
        y_pred_intensities,
        y_true_phases,
        y_pred_phases,
        projector,
        title=f"Intensity to Phase Projection",
        save_path=output_dir / f"intensity_projection_{split}.png",
    )
    plt.close(fig)
    if y_pred_intensities is not None and y_true_intensities is not None:
        log_fucci_color_comparison(
            y_true_intensities,
            y_pred_intensities,
            prefix=f"{split}_fucci_color_comparison",
            save_path=output_dir / f"fucci_color_comparison_{split}.png",
        )
    results["output_dir"] = str(output_dir)
    print(f"\nEvaluation complete! Results saved to: {output_dir}")

    return results


def main():
    """Command-line interface for checkpoint evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate model checkpoint with cyclic metrics")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    parser.add_argument("--data-config", default=None, help="Path to data config YAML")
    parser.add_argument(
        "--split", default="test", choices=["train", "val", "test"], help="Data split to evaluate"
    )
    parser.add_argument("--out", default="logs/checkpoint_eval", help="Output directory")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to evaluate")

    args = parser.parse_args()

    results = evaluate_checkpoint(
        ckpt_path=args.ckpt,
        data_config_path=args.data_config,
        split=args.split,
        output_dir=args.out,
        device=args.device,
        max_samples=args.max_samples,
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(json.dumps(results["metrics"], indent=2))
    print("=" * 60)


if __name__ == "__main__":
    main()
