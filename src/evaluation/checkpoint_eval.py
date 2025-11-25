"""
Unified checkpoint evaluation using the centralized Evaluator.

Provides comprehensive evaluation including:
- Standard regression metrics (R², MAE, MSE, RMSE)
- Geodesic metrics for phase predictions (geodesic MAE, MSE, R²)
- Rich visualizations via PlotGenerator
- Uses same evaluation logic as training for consistency
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.utils import load_ckpt_artifacts, get_device
from src.evaluation.evaluator import Evaluator, EvaluationConfig


def evaluate_checkpoint(
    checkpoint_path: Path,
    eval_config: Optional[EvaluationConfig] = None,
    device: str = "cpu",
    save_dir: Optional[Path] = None,
    split: str = "test",
) -> Dict[str, Any]:
    """
    Evaluate a model checkpoint using the Evaluator.
    Args:
        checkpoint_path: Path to model checkpoint
        eval_config: Configuration for evaluation (plots, metrics, etc.)
        device: Device to run evaluation on
        save_dir: Optional directory to save results

    Returns:
        Dictionary with metrics and plots
    """

    # Load model
    print(f"Loading checkpoint: {checkpoint_path}")
    model, dataloader, projector = load_ckpt_artifacts(
        ckpt_path=checkpoint_path, device=device, split=split
    )
    # Collect predictions
    all_preds = []
    all_targets = []
    all_targets_raw = []  # Store raw 2D intensities for phase models

    print(f"Evaluating...")

    # Temporarily disable projector to get raw intensities
    dataset_projector_backup = None
    if hasattr(dataloader.dataset, "dataset") and hasattr(dataloader.dataset.dataset, "projector"):
        # Handle Subset case
        dataset_projector_backup = dataloader.dataset.dataset.projector
        dataloader.dataset.dataset.projector = None
    elif hasattr(dataloader.dataset, "projector"):
        dataset_projector_backup = dataloader.dataset.projector
        dataloader.dataset.projector = None

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting predictions"):
            x, y = batch

            # Handle tuple/list inputs (image + features)
            if isinstance(x, (list, tuple)):
                x = [item.to(device) if isinstance(item, torch.Tensor) else item for item in x]
                preds = model(*x)
            else:
                x = x.to(device)
                preds = model(x)

            all_preds.append(preds.cpu().numpy())
            all_targets_raw.append(y.numpy() if isinstance(y, torch.Tensor) else y)

    # Restore projector
    if dataset_projector_backup is not None:
        if hasattr(dataloader.dataset, "dataset"):
            dataloader.dataset.dataset.projector = dataset_projector_backup
        else:
            dataloader.dataset.projector = dataset_projector_backup

    y_pred = np.concatenate(all_preds)
    y_true_raw = np.concatenate(all_targets_raw)  # Always 2D intensities

    print(f"Collected {len(y_true_raw)} predictions")
    print(f"Shape - predictions: {y_pred.shape}, raw targets: {y_true_raw.shape}")

    # Determine evaluation mode based on prediction shape
    pred_is_1d = y_pred.ndim == 1 or (y_pred.ndim > 1 and y_pred.shape[1] == 1)
    pred_is_2d = y_pred.ndim > 1 and y_pred.shape[1] == 2

    # Case 1: Model predicts 2D intensities
    # Evaluate in intensity space AND project to phase space for geodesic metrics
    if pred_is_2d:
        print("\n" + "=" * 60)
        print("MODE: 2D Intensity Prediction")
        print("Evaluating in both intensity and phase space")
        print("=" * 60 + "\n")

        # Create evaluation config if not provided
        if eval_config is None:
            plots_to_generate = [
                "scatter",
                "residual",
                "error_heatmap",
                "true_residuals",
                "fucci_color_comparison",
            ]
            eval_config = EvaluationConfig(
                is_phase=False,
                plots_to_generate=plots_to_generate,
            )

        # Evaluate in intensity space
        evaluator_intensity = Evaluator(config=eval_config, projector=projector)
        result_intensity = evaluator_intensity.evaluate(
            y_true_raw, y_pred, prefix=f"{split}/intensity"
        )

        # Project to phase and evaluate with geodesic metrics
        # Project each sample individually
        y_true_phase = np.array([projector.project(sample)[0] for sample in y_true_raw])
        y_pred_phase = np.array([projector.project(sample)[0] for sample in y_pred])
        eval_config_phase = EvaluationConfig(
            is_phase=True,
            plots_to_generate=["geodesic_residual", "phase_scatter"],
        )
        evaluator_phase = Evaluator(config=eval_config_phase, projector=projector)
        result_phase = evaluator_phase.evaluate(y_true_phase, y_pred_phase, prefix=f"{split}/phase")

        # Combine metrics and plots
        metric_dict = result_intensity.metrics.to_dict()

        all_plots = {**result_intensity.plots, **result_phase.plots}

    # Case 2: Model predicts 1D phase
    # We have y_true_raw (2D intensities), project both to phase for geodesic evaluation
    # Then visualize phase errors overlaid on intensity space
    elif pred_is_1d:
        print("\n" + "=" * 60)
        print("MODE: 1D Phase Prediction")
        print("Evaluating in phase space with intensity visualization")
        print("=" * 60 + "\n")

        # Project true intensities to phase (sample by sample)
        print("Projecting ground truth intensities to phase...")
        y_true_phase = np.array([projector.project(sample)[0] for sample in y_true_raw])

        # Flatten predictions if needed
        if y_pred.ndim > 1:
            y_pred_phase = y_pred.flatten()
        else:
            y_pred_phase = y_pred

        # Create evaluation config if not provided
        if eval_config is None:
            plots_to_generate = [
                "scatter",
                "geodesic_residual",
            ]
            eval_config = EvaluationConfig(
                is_phase=True,
                plots_to_generate=plots_to_generate,
            )

        # Evaluate in phase space with geodesic metrics
        evaluator_phase = Evaluator(config=eval_config, projector=projector)
        result_phase = evaluator_phase.evaluate(y_true_phase, y_pred_phase, prefix=f"{split}/phase")

        metric_dict = result_phase.metrics.to_dict()
        all_plots = result_phase.plots

        evaluator_phase.config.plots_to_generate = [
            "true_residuals",
        ]
        residuals = [
            projector.geodesic_distance(y_t, y_p) for y_t, y_p in zip(y_true_phase, y_pred_phase)
        ]

        all_plots.update(
            evaluator_phase._generate_plots(
                y_true=y_true_raw, y_pred=None, residuals=residuals, prefix=f"{split}/phase"
            )
        )
    else:
        raise ValueError(f"Unsupported prediction shape: {y_pred.shape}")

    # Print metrics
    print("\n" + "=" * 60)
    print("METRICS")
    print("=" * 60)
    for key, value in metric_dict.items():
        if value is None:
            continue
        print(f"{key}: {value:.6f}")
    print("=" * 60 + "\n")

    # Save results if directory provided
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics
        metrics_path = save_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metric_dict, f, indent=2)
        print(f"Saved metrics to {metrics_path}")

        # Save plots
        for plot_name, fig in all_plots.items():
            fig.savefig(save_dir / f"{plot_name}.png", dpi=300, bbox_inches="tight")
        print(f"Saved {len(all_plots)} plots to {save_dir}")

    return {
        "metrics": metric_dict,
        "plots": all_plots,
        "num_samples": len(y_true_raw),
    }


def main():
    """Command-line interface for checkpoint evaluation."""
    import argparse
    from src.evaluation.utils import maybe_build_dataloader_from_ckpt

    parser = argparse.ArgumentParser(description="Evaluate model checkpoint with unified evaluator")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--split", default="test", choices=["train", "val", "test"], help="Data split to evaluate"
    )
    parser.add_argument("--out", default="logs/checkpoint_eval", help="Output directory")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    parser.add_argument(
        "--plots",
        nargs="+",
        default=None,
        help="Plots to generate (scatter, residual, error_heatmap, etc.)",
    )
    parser.add_argument(
        "--phase", action="store_true", help="Use phase metrics (for phase predictions)"
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.out) / Path(args.ckpt).stem / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)

    # Create evaluation config
    eval_config = None
    if args.plots or args.phase:
        plots = args.plots if args.plots else ["scatter", "residual"]
        eval_config = EvaluationConfig(
            is_phase=args.phase,
            plots_to_generate=plots,
        )

    # Evaluate
    results = evaluate_checkpoint(
        checkpoint_path=Path(args.ckpt),
        eval_config=eval_config,
        device=device,
        save_dir=output_dir,
        split=args.split,
    )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Evaluated {results['num_samples']} samples")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
