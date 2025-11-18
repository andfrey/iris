from pathlib import Path
from typing import Optional, Callable
import torch
import numpy as np
from typing import Dict
from torch import nn

from src.models.cnet import CNet
from src.data_pipeline.dataset import ModularCellDataModule


def get_device(device: Optional[str] = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_model_from_checkpoint(
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
    dev = get_device(device)

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


def maybe_build_dataloader_from_ckpt(
    ckpt_path: str,
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


def circular_distance(angles1: np.ndarray, angles2: np.ndarray) -> np.ndarray:
    """
    Compute the shortest angular distance between two sets of angles.

    Args:
        angles1: First set of angles in radians
        angles2: Second set of angles in radians

    Returns:
        Angular distances in radians, always in [0, π]
    """
    diff = np.abs(angles1 - angles2)
    return np.minimum(diff, 2 * np.pi - diff)


def circular_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error for circular data (phases)."""
    return float(np.mean(circular_distance(y_true, y_pred)))


def circular_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error for circular data (phases)."""
    return float(np.mean(circular_distance(y_true, y_pred) ** 2))


def circular_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² score for circular data using circular distance."""
    ss_res = np.sum(circular_distance(y_true, y_pred) ** 2)
    # For circular data, compare to mean direction
    mean_direction = np.mean(y_true)
    ss_tot = np.sum(circular_distance(y_true, mean_direction) ** 2)
    return float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0


def evaluate_regression_cyclic(
    y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "test", is_phase: bool = True
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

    # Flatten if needed
    y_true_flat = y_true.flatten() if y_true.ndim > 1 and y_true.shape[1] == 1 else y_true
    y_pred_flat = y_pred.flatten() if y_pred.ndim > 1 and y_pred.shape[1] == 1 else y_pred

    # Standard metrics
    metrics[f"{prefix}_mse"] = mean_squared_error(y_true_flat, y_pred_flat)
    metrics[f"{prefix}_mae"] = mean_absolute_error(y_true_flat, y_pred_flat)
    metrics[f"{prefix}_r2"] = r2_score(y_true_flat, y_pred_flat)
    metrics[f"{prefix}_rmse"] = np.sqrt(metrics[f"{prefix}_mse"])

    # Cyclic metrics for phase data
    if is_phase and y_true_flat.ndim == 1:
        metrics[f"{prefix}_circular_mae"] = circular_mae(y_true_flat, y_pred_flat)
        metrics[f"{prefix}_circular_mse"] = circular_mse(y_true_flat, y_pred_flat)
        metrics[f"{prefix}_circular_r2"] = circular_r2(y_true_flat, y_pred_flat)
        metrics[f"{prefix}_circular_rmse"] = np.sqrt(metrics[f"{prefix}_circular_mse"])

    return metrics
