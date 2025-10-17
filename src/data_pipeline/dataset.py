"""
Modern PyTorch Dataset implementation using the new modular architecture.
"""

import importlib
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import numpy as np
import lightning as L
import yaml
from functools import partial

from .data_sources import DataSource, H5DataSource, FilteredDataSource, CellData
from .data_transforms import Transform, TransformPipeline
from .data_filters import (
    PlaneCountFilter,
    EmptySegmentationFilter,
    MultipleObjectsFilter,
    NucleiSizeFilter,
)


def resolve(name: str):
    module_name, attr_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


class ModularCellDataModule(L.LightningDataModule):
    """
    Lightning DataModule for the cell dataset.

    Example config:
        data:
          h5_path: data/file.h5
          batch_size: 64
          num_workers: 0
          train_val_split: [0.8, 0.2]
          filters:
            plane_count: 3
            max_objects: 1
            min_seg_pixels: 10
            max_nuclei_ratio: 1.2
          transforms:
            plane_selection: middle
            normalize_method: minmax
            gaussian_sigma: 1.0
            channel_order: [bf, "405"]
    """

    def __init__(
        self,
        data_config_path: str = "configs/data_config.yaml",
    ):
        """
        Args:
            data_config_path: Path to YAML config file with data parameters
                            h5_path: Path to H5 file
                            batch_size: Batch size for dataloaders
                            num_workers: Number of workers for dataloaders
                            train_val_split: Train/validation split ratios [train, val]

                            Filter parameters:
                            use_quality_filters: Whether to apply quality filters
                            plane_count: Expected number of planes per channel
                            max_objects: Maximum objects in segmentation mask
                            min_seg_pixels: Minimum pixels in segmentation mask
                            max_nuclei_ratio: Maximum nuclei/cell area ratio
                            force_refilter: Force re-filtering even if cache exists

                            Transform parameters:
                            plane_selection: Plane selection strategy ('middle', 'first', 'last', 'all')
                            normalize_method: Normalization method ('minmax' or 'percentile')
                            normalize_channels: Channels to normalize
                            apply_gaussian: Whether to apply Gaussian smoothing
                            gaussian_sigma: Sigma for Gaussian filter
                            gaussian_channels: Channels to apply Gaussian filter to
                            channel_order: Order of channels in output
                            seed: Random seed for reproducibility
        """
        super().__init__()
        with open(data_config_path, "r") as f:
            self.config = yaml.safe_load(f)
        # Save hyperparameters for Lightning
        self.save_hyperparameters(self.config)

        # Store parameters
        self.h5_path = self.config.get("h5_path", None)
        self.batch_size = self.config.get("batch_size", 64)
        self.num_workers = self.config.get("num_workers", 0)
        self.train_val_split = self.config.get("train_val_split", [0.8, 0.2])
        self.seed = self.config.get("seed", 42)  # For reproducibility

        # Filter parameters
        self.use_quality_filters = self.config.get("use_quality_filters", True)
        self.plane_count = self.config.get("plane_count", 3)
        self.max_objects = self.config.get("max_objects", 1)
        self.min_seg_pixels = self.config.get("min_seg_pixels", 10)
        self.max_nuclei_ratio = self.config.get("max_nuclei_ratio", 1.2)
        self.force_refilter = self.config.get("force_refilter", False)

        # Transform parameters
        self.transform_configs = self.config.get("transforms", [])

        # Datasets (initialized in setup)
        self.train_dataset = None
        self.val_dataset = None
        self.full_dataset = None

    def prepare_data(self):
        """Check if data exists"""
        if self.h5_path is None:
            raise ValueError("h5_path must be specified in data config")
        h5_path = Path(self.h5_path)
        if not h5_path.exists():
            raise FileNotFoundError(f"H5 file not found: {self.h5_path}")

        print(f"✓ H5 file found: {self.h5_path}")

    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for each stage (fit, validate, test, predict).

        Args:
            stage: Current stage ('fit', 'validate', 'test', 'predict', or None)
        """
        # Only setup once
        if self.full_dataset is not None:
            return

        print("\n" + "=" * 60)
        print("Setting up ModularCellDataModule")
        print("=" * 60)

        # 1. Create data source
        print(f"\n1. Creating data source from: {self.h5_path}")
        data_source = H5DataSource(
            path=self.h5_path,
            plane_selection="all",  # Load all planes, select later in transform
        )
        print(f"   ✓ Found {len(data_source.get_cell_ids()):,} total cells")

        # 2. Apply quality filters (optional)
        if self.use_quality_filters:
            print(f"\n2. Applying quality filters:")
            print(f"   - Plane count: {self.plane_count}")
            print(f"   - Max objects: {self.max_objects}")
            print(f"   - Min segmentation pixels: {self.min_seg_pixels}")
            print(f"   - Max nuclei/cell ratio: {self.max_nuclei_ratio}")

            filters = [
                PlaneCountFilter(expected_planes=self.plane_count),
                EmptySegmentationFilter(min_pixels=self.min_seg_pixels),
                MultipleObjectsFilter(max_objects=self.max_objects),
                NucleiSizeFilter(max_ratio=self.max_nuclei_ratio),
            ]

            filtered_source = FilteredDataSource(
                data_source=data_source,
                filters=filters,
                cache_results=True,
                show_progress=True,
                force_refilter=self.force_refilter,
            )

            print(
                f"   ✓ Filtered to {len(filtered_source.get_cell_ids()):,} valid cells"
            )

            data_source = filtered_source
        else:
            print(f"\n2. Skipping quality filters")

        # 3. Create transform pipeline
        print(f"\n3. Creating preprocessing pipeline:")
        transforms = [
            resolve(cfg["class_path"])(**cfg.get("init_args", {}))
            for cfg in self.transform_configs
        ]
        transform_pipeline = TransformPipeline(transforms)

        # 4. Create full dataset
        print(f"\n4. Creating PyTorch dataset")
        self.full_dataset = ModularCellDataset(
            data_source=data_source, transform=transform_pipeline
        )

        # 5. Split into train/val
        if stage == "fit" or stage is None:
            print(f"\n5. Splitting dataset: {self.train_val_split}")

            # Calculate split sizes
            total_size = len(self.full_dataset)
            train_size = int(total_size * self.train_val_split[0])
            val_size = np.ceil((total_size - train_size) / 2).astype(int)
            test_size = (
                total_size - val_size - train_size
            )  # Adjust train size to use all samples

            assert (
                sum([train_size, val_size, test_size]) == total_size
            ), "Train/val/test sizes do not sum to total"
            # Split with generator for reproducibility
            generator = torch.Generator().manual_seed(self.seed)
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                self.full_dataset,
                [train_size, val_size, test_size],
                generator=generator,
            )

            print(f"   ✓ Train: {len(self.train_dataset):,} samples")
            print(f"   ✓ Val:   {len(self.val_dataset):,} samples")
            print(f"   ✓ Test:  {len(self.test_dataset):,} samples")

        print("\n" + "=" * 60)
        print("✓ DataModule setup complete!")
        print("=" * 60 + "\n")

    def train_dataloader(self) -> DataLoader:
        """Returns training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self) -> DataLoader:
        """Returns test dataloader (uses val dataset for now)."""
        return self.val_dataloader()

    def predict_dataloader(self) -> DataLoader:
        """Returns prediction dataloader (full dataset)."""
        return DataLoader(
            self.full_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.num_workers > 0 else False,
        )


class ModularCellDataset(Dataset):
    """
    Generic cell dataset that works with any DataSource and Transform pipeline.

    This is a simpler, more modular alternative to the original CellImageDataset.
    """

    def __init__(
        self,
        data_source: DataSource,
        transform: Optional[Transform] = None,
        mask_intensity: str = "segment",
        debug: bool = False,
    ):
        """
        Args:
            data_source: DataSource instance (can be FilteredDataSource)
            transform: Transform pipeline for preprocessing
        """
        self.data_source = data_source
        self.transform = transform
        self.mask_intensity = mask_intensity
        self.debug = debug
        # Get cell IDs
        self.cell_ids = data_source.get_cell_ids()
        print(f"✓ Dataset initialized with {len(self.cell_ids):,} cells")

    def __len__(self) -> int:
        return len(self.cell_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Tuple of (images, labels) where:
            - images: Tensor of shape (C, H, W)
            - labels: Tensor of shape (2,) with [488_intensity, 561_intensity]
        """
        cell_id = self.cell_ids[idx]
        cell_data = self.data_source.load_cell(cell_id)

        # Compute FUCCI labels (488 and 561 intensities)
        labels = compute_fucci_labels(
            cell_data, mask_intensity=self.mask_intensity, debug=self.debug
        )

        # Apply transforms
        if self.transform:
            cell_data = self.transform(cell_data)

        channel_list = []
        for channel_name in ["bf", "405"]:
            for plane in cell_data.channels.get(channel_name, None):
                if plane is None:
                    raise ValueError(
                        f"Missing channel {channel_name} for cell {cell_id}"
                    )
                channel_list.append(plane)
        images = np.array(channel_list)

        # Ensure float32 and proper shape (C, H, W)
        images = images.astype(np.float32)
        if images.ndim == 2:
            images = images[np.newaxis, :]  # Add channel dim

        # Convert to tensors
        images_tensor = torch.from_numpy(images)
        labels_tensor = torch.from_numpy(labels).float()

        return images_tensor, labels_tensor


def compute_fucci_labels(
    cell_data: CellData, mask_intensity: str = "segmentation", debug: bool = False
) -> np.ndarray:
    """
    Compute FUCCI mean log intensities (488 and 561) from cell data within the specified mask.

    Returns:
        Array of shape (2,) with [intensity_488, intensity_561]
    """
    labels = []
    if mask_intensity not in ["segmentation", "nuclei_segmentation"]:
        raise ValueError("mask_intensity must be 'segmentation' or 'nuclei_segmentation'")
    # Only compute mean intensity within the mask_intensity region
    if not hasattr(cell_data, mask_intensity):
        # No mask available, fall back to computing over entire image
        raise ValueError("Missing segmentation mask for intensity computation")

    # Compute intensity only within the masked region
    labels = []
    for channel in ["488", "561"]:
        if channel in cell_data.channels:
            planes = cell_data.channels[channel]
            masks = getattr(cell_data, mask_intensity)
            if masks is None:
                raise ValueError(
                    f"Missing mask '{mask_intensity}' for intensity computation"
                )
            if len(planes) != len(masks):
                raise ValueError(
                    f"Channel {channel} planes and segmentation planes count mismatch"
                )
            # Compute mean intensity only within mask for each plane
            intensities = []
            for plane, mask in zip(planes, masks):
                mean_outside_mask = plane[mask == 0].mean()
                plane_normalized = plane - mean_outside_mask  # Background subtraction
                plane_normalized = np.clip(
                    plane_normalized, 0, None
                )  # Remove negatives
                if debug:
                    show_images(
                        [plane, mask, plane_normalized],
                        titles=[f"{channel} plane", "Mask", "Normalized"],
                    )
                masked_values = plane_normalized[mask > 0]
                if len(masked_values) > 0:
                    masked_values_mean = masked_values.mean()
                    if np.isnan(masked_values_mean):
                        masked_values_mean = 0.0
                    intensities.append(masked_values_mean)

            if not intensities:
                raise ValueError(f"No valid intensities found for channel {channel}")
            mean_intensity = np.mean(intensities)
            assert mean_intensity <= max(
                [plane.max() for plane in planes]
            ), "Mean intensity exceeds max plane intensity"
            log_mean_intensity = (
                np.log(mean_intensity) if mean_intensity > 0.0 else 0.0
            )  # Log normalization
            labels.append(log_mean_intensity)
        else:
            raise ValueError(f"Channel {channel} not found in cell data")

    return np.array(labels, dtype=np.float32)


def show_images(images: List[np.ndarray], titles: Optional[List[str]] = None):
    import matplotlib.pyplot as plt

    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(15, 5))
    if n == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles or []):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    plt.show()
