from pathlib import Path
from typing import Optional, Callable, Tuple
import torch
import numpy as np
from typing import Dict
from torch import nn


from src.models.cnet import CNet
from src.data_pipeline.dataset import ModularCellDataModule
from src.data_pipeline.curve_projector import FucciCurveProjector


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


def load_ckpt_artifacts(
    ckpt_path: str, split: str = "val", device: Optional[str] = None
) -> Tuple[nn.Module, torch.utils.data.DataLoader, FucciCurveProjector]:
    """
    Load data and model config artifacts from checkpoint if available.

    Args:
        ckpt_path: Path to model checkpoint
    """

    print(f"Loading checkpoint from: {ckpt_path}")

    # Load model
    model = load_model_from_checkpoint(ckpt_path)
    model.eval().to(device)

    loader = maybe_build_dataloader_from_ckpt(ckpt_path, split)
    if loader is None:
        print("Failed to create dataloader from checkpoint.")
        return
    dataset = loader.dataset.dataset
    if dataset.projector is not None:
        projector = dataset.projector
    else:
        projector = FucciCurveProjector(dataset=dataset)
        projector.fit_from_dataset(use_cache=True)

    return model, loader, projector
