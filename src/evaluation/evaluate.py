"""
Evaluation helpers to explain CNN checkpoints with Captum and summarize FC-layer feature importance.

Provides a callable `analyze_checkpoint_with_captum` and a small CLI.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Optional, Tuple, Dict, Any, List
import inspect

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Ensure repo root is importable when running as a script
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.cnet import CNet  # type: ignore
from src.data_pipeline.dataset import ModularCellDataModule  # type: ignore
import yaml


def _get_device(device: Optional[str] = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_model_from_checkpoint(
    ckpt_path: str,
    model_factory: Optional[Callable[[], nn.Module]] = None,
    device: Optional[str] = None,
) -> nn.Module:
    """
    Load a model from a checkpoint.

    - If `model_factory` is provided, it should create an uninitialized model; the state_dict
      will be loaded from the checkpoint (assumed to be a regular torch checkpoint with 'state_dict').
    - Otherwise, attempt to load using LightningModule.load_from_checkpoint with CNet.
    """
    dev = _get_device(device)

    if model_factory is not None:
        model = model_factory()
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        # Strip potential 'model.' or 'net.' prefixes if present
        new_state = {}
        for k, v in state_dict.items():
            new_state[k.replace("model.", "").replace("net.", "")] = v
        model.load_state_dict(new_state, strict=False)
    else:
        # Fallback to Lightning checkpoint for CNet
        model = CNet.load_from_checkpoint(ckpt_path)  # type: ignore[attr-defined]

    model.eval().to(dev)
    return model


def _get_dataloader(
    data_config: dict,
    split: str = "val",
) -> torch.utils.data.DataLoader:
    data_config.update({"num_workers": 0})
    dm = ModularCellDataModule(data_config_path=None, data_config=data_config)
    dm.prepare_data()
    dm.setup()
    if split == "train":
        return dm.train_dataloader()
    if split == "test":
        return dm.test_dataloader()
    if split == "predict":
        return dm.predict_dataloader()
    return dm.val_dataloader()


def _maybe_build_dataloader_from_ckpt(
    ckpt_path: str,
    out_dir: Path,
    split: str = "val",
) -> Optional[torch.utils.data.DataLoader]:
    """
    Try to reconstruct dataloaders using information stored in a Lightning checkpoint.
    Heuristics:
      - Look for 'hyper_parameters' with keys 'data_config_path' or 'data_config'/'data'
      - If a config dict is found, write a temporary YAML and instantiate ModularCellDataModule
    Returns a DataLoader or None if reconstruction fails.
    """

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except Exception:
        return None

    # Case 1: direct path saved
    data_cfg = ckpt.get("datamodule_hyper_parameters")

    return _get_dataloader(data_config=data_cfg, split=split)


def _compute_ig_attributions(
    model: nn.Module,
    inputs: torch.Tensor,
    target_index: Optional[int] = None,
    baseline: str = "zeros",
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Compute Integrated Gradients attributions for a batch of inputs.

    inputs: [B, C, H, W]
    target_index: output index for which to compute attribution (if multi-output)
    baseline: 'zeros' or 'median'
    """
    # Local import to avoid hard dependency during type checking / when unused
    try:
        from captum.attr import IntegratedGradients  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency
        raise ImportError(
            "captum is not installed. Please install captum to use attribution analysis: pip install captum"
        ) from e

    inputs[0].requires_grad = True
    if len(inputs) > 1 and inputs[1] is not None:
        inputs[1].requires_grad = True
    ig = IntegratedGradients(model)

    baselines = (
        torch.zeros_like(inputs)
        if not isinstance(inputs, (tuple, list))
        else tuple(torch.zeros_like(inp) for inp in inputs)
    )
    if baseline == "median":
        # Per-channel median baseline
        med = inputs[0].median(dim=-1, keepdim=True).values.median(dim=-2, keepdim=True).values
        med = med.expand_as(inputs[0])
        if len(inputs) > 1 and inputs[1] is not None:
            med_features = (
                inputs[1].median(dim=-1, keepdim=True).values.median(dim=-2, keepdim=True).values
            )
            med_features = med_features.expand_as(inputs[1])
            med = (med, med_features)
        baselines = tuple(med)

    attributions, delta = ig.attribute(
        inputs,
        baselines=baselines,
        target=target_index,
        return_convergence_delta=True,
        n_steps=32,
    )
    return attributions, delta


def _compute_gradcam(
    model: nn.Module,
    inputs: torch.Tensor,
    target_layer: Optional[nn.Module] = None,
    target_index: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute GradCAM heatmaps for a batch of inputs.

    Args:
        model: The neural network model
        inputs: Input tensor or tuple of tensors (images, features)
        target_layer: Layer to compute GradCAM on. If None, uses last conv layer.
        target_index: Output index for which to compute attribution

    Returns:
        Tensor of GradCAM heatmaps [B, H, W]
    """
    try:
        from captum.attr import LayerGradCam  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "captum is not installed. Please install captum to use GradCAM: pip install captum"
        ) from e

    # Auto-detect last convolutional layer if not specified
    if target_layer is None:
        conv_layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append((name, module))
        if not conv_layers:
            raise ValueError("No Conv2d layers found in model for GradCAM")
        target_layer = conv_layers[-1][1]
        print(f"Using layer for GradCAM: {conv_layers[-1][0]}")

    # Prepare inputs
    if isinstance(inputs, (tuple, list)):
        inp_tensor = inputs[0]
    else:
        inp_tensor = inputs

    inp_tensor.requires_grad = True

    # Compute GradCAM
    gradcam = LayerGradCam(model, target_layer)
    attributions = gradcam.attribute(
        inputs,
        target=target_index,
        relu_attributions=True,  # Apply ReLU to get positive attributions
    )

    return attributions


def _plot_attribution_examples(
    inputs: torch.Tensor,
    attributions: torch.Tensor,
    out_dir: Path,
    max_examples: int = 8,
    channel_titles: Optional[List[str]] = None,
    prefix: str = "ig",
    gradcam_heatmaps: Optional[torch.Tensor] = None,
    predictions: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
) -> None:
    """
    Plot a few examples with per-channel overlays and save to files.
    Expects [B, C, H, W] tensors.

    Args:
        inputs: Input images [B, C, H, W]
        attributions: Attribution maps [B, C, H, W]
        out_dir: Output directory
        max_examples: Number of examples to plot
        channel_titles: Optional channel names
        prefix: Filename prefix
        gradcam_heatmaps: Optional GradCAM heatmaps [B, H, W] to overlay
        predictions: Optional predictions [B, 1] (phase in radians [-pi, pi])
        labels: Optional ground truth labels [B, 1] (phase in radians [-pi, pi])
    """
    _ensure_dir(out_dir)

    b = min(max_examples, inputs.shape[0])
    for i in range(b):
        x = inputs[i].detach().cpu()
        a = attributions[i].detach().cpu()

        C = x.shape[0]
        cols = 3 if gradcam_heatmaps is not None else 2

        # Add extra row for phase plot if predictions available
        extra_rows = 0
        if predictions is not None and predictions.shape[-1] == 1:
            extra_rows = 1

        rows = C + extra_rows
        plt.figure(figsize=(cols * 4, rows * 4))

        for c in range(C):
            title_img = (
                channel_titles[c] if channel_titles and c < len(channel_titles) else f"ch{c}"
            )

            # Original image
            plt.subplot(rows, cols, c * cols + 1)
            plt.imshow(x[c], cmap="gray")
            plt.title(f"{title_img}")
            plt.axis("off")

            # Integrated Gradients
            plt.subplot(rows, cols, c * cols + 2)
            plt.imshow(x[c], cmap="gray")
            heat = a[c]
            heat_np = heat.numpy()
            vmax = max(1e-6, float(abs(heat_np).max()))
            # Create custom colormap: red (negative) -> white (zero) -> green (positive)
            colors = ["darkred", "lightcoral", "white", "lightgreen", "darkgreen"]
            n_bins = 256
            cmap = LinearSegmentedColormap.from_list("red_white_green", colors, N=n_bins)

            plt.imshow(heat, cmap=cmap, alpha=0.6, vmin=-vmax, vmax=vmax)
            plt.title(f"{title_img} IG")
            plt.axis("off")
            plt.colorbar(shrink=0.6)

            # GradCAM (if available)
            if gradcam_heatmaps is not None:
                plt.subplot(rows, cols, c * cols + 3)
                plt.imshow(x[c], cmap="gray")
                gradcam = gradcam_heatmaps[i].detach().cpu()

                # Squeeze out any singleton dimensions
                while gradcam.ndim > 2:
                    if gradcam.shape[0] == 1:
                        gradcam = gradcam.squeeze(0)
                    else:
                        break

                # Upsample GradCAM to match input size if needed
                if gradcam.shape != x[c].shape:
                    # gradcam is [H_g, W_g], need to make it [1, 1, H_g, W_g] for interpolate
                    gradcam = torch.nn.functional.interpolate(
                        gradcam.unsqueeze(0).unsqueeze(0),
                        size=(x[c].shape[0], x[c].shape[1]),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze()
                plt.imshow(gradcam, cmap="jet", alpha=0.6)
                plt.title(f"{title_img} GradCAM")
                plt.axis("off")
                plt.colorbar(shrink=0.6)

        # Add circular phase plot if predictions available
        if predictions is not None and predictions.shape[-1] == 1:
            # Create polar subplot spanning all columns at the bottom row
            ax = plt.subplot(rows, cols, C * cols + 1, projection="polar")

            # Extract phase values (assuming they're in radians [-pi, pi])
            pred_phase = float(predictions[i].detach().cpu().item())

            # Plot the circular phase indicator
            # Draw circle
            theta = torch.linspace(0, 2 * 3.14159, 100)
            r = torch.ones_like(theta)
            ax.plot(theta, r, "k-", linewidth=2, alpha=0.3)

            # Draw prediction arrow
            ax.arrow(
                0,
                0,
                pred_phase,
                1.0,
                head_width=0.1,
                head_length=0.00001,
                fc="blue",
                ec="blue",
                linewidth=2,
                alpha=0.8,
                label="Prediction",
            )

            # Draw ground truth if available
            if labels is not None:
                true_phase = float(labels[i].detach().cpu().item())
                ax.arrow(
                    0,
                    0,
                    true_phase,
                    1.0,
                    head_width=0.1,
                    head_length=0.0001,
                    fc="green",
                    ec="green",
                    linewidth=2,
                    alpha=0.6,
                    label="Ground Truth",
                )

            # Customize polar plot
            ax.set_ylim(0, 1)
            ax.set_theta_zero_location("E")  # 0 radians at east (right)
            ax.set_theta_direction(1)  # Counter-clockwise
            ax.set_title(
                f"Cell Cycle Phase\nPred: {pred_phase:.3f} rad ({pred_phase*180/3.14159:.1f}°)",
                fontsize=10,
                pad=20,
            )
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = out_dir / f"{prefix}_example_{i}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()


def _fc_feature_importance_stats(
    model: nn.Module,
    top_k: int = 20,
) -> Dict[str, Any]:
    """
    Compute simple FC-layer feature importance stats using absolute weight magnitudes.

    Returns a dict with layer name, per-input importance vector, summary stats, and top-k indices.
    """
    first_linear: Optional[Tuple[str, nn.Linear]] = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            first_linear = (name, module)
            break

    if first_linear is None:
        return {"has_linear": False}

    name, layer = first_linear
    w = layer.weight.detach().abs().cpu()  # [out_features, in_features]
    importance_per_input = w.mean(dim=0)  # [in_features]
    imp_np = importance_per_input.numpy()

    # Top-k indices
    k = min(top_k, imp_np.shape[0])
    top_idx = imp_np.argsort()[::-1][:k].tolist()
    top_vals = imp_np[top_idx].tolist()

    stats = {
        "has_linear": True,
        "layer": name,
        "in_features": int(layer.in_features),
        "out_features": int(layer.out_features),
        "importance": imp_np.tolist(),
        "mean": float(imp_np.mean()),
        "std": float(imp_np.std()),
        "top_k_indices": top_idx,
        "top_k_values": top_vals,
    }

    # Optionally attach feature names if present on model
    feature_names = getattr(model, "feature_names", None)
    if feature_names and isinstance(feature_names, (list, tuple)):
        stats["top_k_features"] = [feature_names[i] for i in top_idx if i < len(feature_names)]

    return stats


def _save_stats(stats: Dict[str, Any], out_dir: Path, filename: str = "fc_importance.json") -> None:
    _ensure_dir(out_dir)
    with (out_dir / filename).open("w") as f:
        json.dump(stats, f, indent=2)


def analyze_checkpoint_with_captum(
    ckpt_path: str,
    data_config_path: Optional[str] = None,
    output_dir: str = "logs/captum_eval",
    split: str = "val",
    target_index: Optional[int] = None,
    max_examples: int = 8,
    device: Optional[str] = None,
    model_factory: Optional[Callable[[], nn.Module]] = None,
    use_gradcam: bool = True,
    gradcam_layer: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze a CNN checkpoint with Captum and produce visual examples and FC-layer stats.

    Args:
        ckpt_path: Path to model checkpoint file.
        data_config_path: Path to data YAML used by ModularCellDataModule.
        output_dir: Where to save figures and stats.
        split: Data split to sample from ('train', 'val', 'test', 'predict').
        target_index: Output index to attribute. If None and model outputs >1 dim, index 0 is used.
        max_examples: Number of example plots to save.
        device: Explicit device string, else auto-select.
        model_factory: Optional callable returning an nn.Module to load state_dict into.
        use_gradcam: Whether to compute GradCAM attributions in addition to IG.
        gradcam_layer: Name of layer for GradCAM. If None, uses last conv layer.

    Returns:
        A dictionary with summary info including the path to outputs and FC stats.
    """
    out_dir = Path(output_dir)
    _ensure_dir(out_dir)
    dev = _get_device(device)

    # Load model
    model = _load_model_from_checkpoint(ckpt_path, model_factory=model_factory, device=device)

    loader = _maybe_build_dataloader_from_ckpt(ckpt_path, out_dir=out_dir, split=split)
    if loader is None:
        raise ValueError(
            "Could not infer data config from checkpoint. Please pass data_config_path explicitly."
        )
    batch = next(iter(loader))
    inputs = (batch[0][0], batch[0][1]) if isinstance(batch[0], (list, tuple)) else (batch[0],)

    # Extract labels if available
    labels = batch[1] if len(batch) > 1 else None

    inputs = (inputs[0][0:10],)
    if labels is not None:
        labels = labels[0:10]

    # Determine target index if not provided and get predictions
    with torch.no_grad():
        predictions = model(*inputs)

    # Compute attributions
    attributions, delta = _compute_ig_attributions(model, inputs, target_index=target_index)

    # Optionally compute GradCAM
    gradcam_heatmaps = None
    if use_gradcam:
        try:
            # Find target layer for GradCAM
            target_layer = None
            if gradcam_layer:
                # Use specified layer
                for name, module in model.named_modules():
                    if name == gradcam_layer:
                        target_layer = module
                        break
            else:
                # Auto-detect last Conv2d layer
                for name, module in model.named_modules():
                    if isinstance(module, nn.Conv2d):
                        target_layer = module

            if target_layer is not None:
                gradcam_heatmaps = _compute_gradcam(
                    model, inputs, target_layer, target_index=target_index
                )
            else:
                print("Warning: No Conv2d layer found for GradCAM, skipping.")
        except Exception as e:
            print(f"Warning: GradCAM computation failed: {e}")

    # Plot examples
    ds = getattr(loader, "dataset", None)
    # Unwrap Subset if necessary
    base_ds = getattr(ds, "dataset", ds)
    channel_titles = getattr(base_ds, "input_channels", None)
    _plot_attribution_examples(
        inputs[0],
        attributions[0],
        out_dir,
        max_examples=max_examples,
        channel_titles=channel_titles,
        gradcam_heatmaps=gradcam_heatmaps,
        predictions=predictions,
        labels=labels,
    )

    # FC stats
    fc_stats = _fc_feature_importance_stats(model)
    _save_stats(fc_stats, out_dir, filename="fc_importance.json")

    summary = {
        "output_dir": str(out_dir),
        "target_index": target_index,
        "has_linear": bool(fc_stats.get("has_linear", False)),
        "num_examples_plotted": min(max_examples, int(inputs[0].shape[0])),
        "gradcam_computed": gradcam_heatmaps is not None,
    }
    return summary


def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(description="Analyze CNN checkpoint with Captum")
    p.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    p.add_argument("--data-config", default=None, required=False, help="Path to data yaml config")
    p.add_argument("--out", default="logs/captum_eval", help="Output directory")
    p.add_argument("--split", default="val", choices=["train", "val", "test", "predict"])
    p.add_argument("--target", type=int, default=None, help="Target output index for attribution")
    p.add_argument("--examples", type=int, default=8, help="Number of examples to plot")
    p.add_argument("--device", default=None, help="Device e.g. cuda, cuda:0, or cpu")
    p.add_argument("--no-gradcam", action="store_true", help="Disable GradCAM computation")
    p.add_argument(
        "--gradcam-layer",
        default=None,
        help="Specific layer name for GradCAM (default: last conv layer)",
    )
    return p


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    summary = analyze_checkpoint_with_captum(
        ckpt_path=args.ckpt,
        data_config_path=args.data_config,
        output_dir=args.out,
        split=args.split,
        target_index=args.target,
        max_examples=args.examples,
        device=args.device,
        use_gradcam=not args.no_gradcam,
        gradcam_layer=args.gradcam_layer,
    )
    print(json.dumps(summary, indent=2))
