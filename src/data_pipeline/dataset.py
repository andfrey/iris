"""
Modern PyTorch Dataset implementation using the new modular architecture.
"""

from dataclasses import dataclass, field
import yaml
from typing import Optional, List, Dict, Any, Tuple, Union
from copy import deepcopy
import traceback

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import v2
from pathlib import Path
import numpy as np
import lightning as L
from tqdm import tqdm
import pandas as pd


from .utils import (
    save_parquet_cache,
    load_parquet_cache,
    make_hash_from_dict,
)
from .data_sources import DataSource, CellData, DataSourceConfig, create_data_source_from_config
from .data_transforms import Transform, create_transform_pipeline_from_config
from .feature_extractor import FeatureExtractor
from .curve_projector import FucciCurveProjector


@dataclass
class DataModuleConfig:
    batch_size: int = 64
    num_workers: int = 16
    data_split: List[float] = field(default_factory=lambda: [0.7, 0.2, 0.1])
    seed: int = 42


@dataclass
class DataSetConfig(DataModuleConfig):
    data_source: Optional[DataSourceConfig] = None
    image_transforms: List[Dict[str, Any]] = field(default_factory=list)
    feature_transforms: List[Dict[str, Any]] = field(default_factory=list)
    augmentations: Optional[Dict[str, Any]] = None
    input_channels: List[str] = field(default_factory=lambda: ["bf", "405"])
    add_features: bool = False
    mask_intensity: str = "segmentation"
    fucci_transform: Dict[str, Any] = field(default_factory=dict)
    fucci_1D_projection: bool = False
    projection_shape: Optional[str] = None

    def __post_init__(self):
        """Convert dict to DataSourceConfig if needed."""
        if isinstance(self.data_source, dict):
            self.data_source = DataSourceConfig(**self.data_source)

    def to_dict(self) -> Dict[str, Any]:
        """Convert DataSetConfig to dictionary for legacy compatibility."""
        from dataclasses import asdict

        return asdict(self)


# === BaseCellDataset ===
class BaseCellDataset(Dataset):
    def __init__(
        self,
        data_config: DataSetConfig,
    ):
        if isinstance(data_config, dict):
            self.config = DataSetConfig(**data_config)
        else:
            self.config = data_config

        self.image_transform = create_transform_pipeline_from_config(
            self.config.image_transforms, transform_type="image"
        )
        self.mask_intensity = self.config.mask_intensity

        # Create data source
        self.data_source = create_data_source_from_config(self.config.data_source)

        # Handle FUCCI transforms
        if "scale" in self.config.fucci_transform:
            self.scale_divider_488 = self.config.fucci_transform.get("scale_divider_488", 1.0)
            self.scale_divider_561 = self.config.fucci_transform.get("scale_divider_561", 1.0)
            self.fucci_scaler = np.array([self.scale_divider_488, self.scale_divider_561])
            print(f"✓ Scale dividers: 488={self.scale_divider_488}, 561={self.scale_divider_561}")

        self.cell_ids = self.data_source.get_cell_ids()
        self.fucci_1D_projection = self.config.fucci_1D_projection
        self.projector = None

        if self.fucci_1D_projection:
            self.projector = FucciCurveProjector(
                dataset=deepcopy(self), shape=self.config.projection_shape
            )
            self.projector.fit_from_dataset(use_cache=True)
            print("✓ FUCCI 1D projection enabled")

        print(f"✓ Dataset initialized with {len(self.cell_ids):,} cells")

    def __len__(self) -> int:
        return len(self.cell_ids)

    def get_cell_data_fucci_labels(self, idx: int):
        cell_id = self.cell_ids[idx]
        cell_data = self.data_source.load_cell(cell_id)
        labels = self.compute_labels(cell_data)
        if self.fucci_1D_projection and self.projector is not None:
            labels = self.projector.project(labels)[0]
            labels = np.array([labels]) if isinstance(labels, float) else np.array(labels)
        if self.image_transform:
            cell_data = self.image_transform(cell_data)
        return cell_data, labels

    def compute_labels(self, cell_data):
        labels = compute_fucci_labels(cell_data, mask_intensity=self.mask_intensity)

        if "scale" in self.config.fucci_transform:
            labels = labels / self.fucci_scaler
        elif "log" in self.config.fucci_transform:
            # to avoid log(0) and log of negative values
            if labels[0] <= 1e-6:
                labels[0] = 1e-6
            if labels[1] <= 1e-6:
                labels[1] = 1e-6
            labels = np.log(labels)

        return labels

    def get_projector(self) -> FucciCurveProjector:
        return self.projector


class ModularCellDataModule(L.LightningDataModule):
    """
    Lightning DataModule for the cell dataset.
    """

    def __init__(
        self,
        data_config_path: Optional[str] = None,
        data_config: Optional[DataSetConfig] = None,
    ):
        """
        Args:
            data_config_path: Path to YAML config file
            data_config: DataSetConfig object (alternative to yaml path)
        """
        super().__init__()

        if data_config is None:
            if data_config_path is None:
                raise ValueError("Either data_config_path or data_config must be provided")
            with open(data_config_path, "r") as f:
                config_dict = yaml.safe_load(f)
            self.config = DataSetConfig(**config_dict)
        elif isinstance(data_config, dict):
            self.config = DataSetConfig(**data_config)
        else:
            self.config = data_config

        # Save hyperparameters for Lightning
        self.save_hyperparameters(self.config.to_dict())

        # Datasets (initialized in setup)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.full_dataset = None

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

        if self.config.add_features:
            self.full_dataset = ModularCellImageFeatureDataset(self.config)
        else:
            self.full_dataset = ModularCellImageDataset(self.config)

        # Split with generator for reproducibility
        self.train_dataset, self.val_dataset, self.test_dataset = split_dataset(
            self.full_dataset, self.config.data_split, self.config.seed
        )
        self.train_dataset = torch.utils.data.Subset(
            deepcopy(self.full_dataset), self.train_dataset.indices
        )
        self.val_dataset = torch.utils.data.Subset(
            deepcopy(self.full_dataset), self.val_dataset.indices
        )
        self.test_dataset = torch.utils.data.Subset(
            deepcopy(self.full_dataset), self.test_dataset.indices
        )

        print(f"   ✓ Train: {len(self.train_dataset):,} samples")
        print(f"   ✓ Val:   {len(self.val_dataset):,} samples")
        print(f"   ✓ Test:  {len(self.test_dataset):,} samples")

        # Enable augmentation only on training dataset (if available in the dataset)
        self.train_dataset.dataset.apply_augmentation = True
        self.val_dataset.dataset.apply_augmentation = False
        self.test_dataset.dataset.apply_augmentation = False

        print("✓ Augmentation applied to training dataset")

        print("\n" + "=" * 60)
        print("✓ DataModule setup complete!")
        print("=" * 60 + "\n")

    def train_dataloader(self) -> DataLoader:
        """Returns training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True if self.config.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True if self.config.num_workers > 0 else False,
        )

    def test_dataloader(self) -> DataLoader:
        """Returns test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True if self.config.num_workers > 0 else False,
        )

    def predict_dataloader(self) -> DataLoader:
        """Returns prediction dataloader (full dataset)."""
        return DataLoader(
            self.full_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True if self.config.num_workers > 0 else False,
        )


class ModularCellImageDataset(BaseCellDataset):
    """
    Generic cell image dataset that works with any DataSource and Transform pipeline.
    """

    def __init__(self, data_config: DataSetConfig):
        if isinstance(data_config, dict):
            data_config = DataSetConfig(**data_config)

        self.input_channels = data_config.input_channels
        # Flag toggled by the DataModule to ensure augmentation is applied only on the train split
        self.apply_augmentation = False
        super().__init__(data_config=data_config)
        # Build an optional torchvision-based augmentation pipeline from config
        self.augmentation_transform = self.create_augmentation_transform()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cell_data, labels = self.get_cell_data_fucci_labels(idx)
        channel_list = []
        for channel_name in self.input_channels:
            if channel_name not in cell_data.channels:
                if getattr(cell_data, channel_name, None):
                    planes = getattr(cell_data, channel_name)
                else:
                    raise ValueError(
                        f"Channel {channel_name} not found for cell {self.cell_ids[idx]}"
                    )
            else:
                planes = cell_data.channels[channel_name]
            for plane in planes:
                if plane is None:
                    raise ValueError(
                        f"Missing channel {channel_name} for cell {self.cell_ids[idx]}"
                    )
                channel_list.append(plane)
        images = np.array(channel_list)
        images = images.astype(np.float32)
        if images.ndim == 2:
            images = images[np.newaxis, :]
        images_tensor = torch.from_numpy(images)
        # Apply augmentation only when enabled for this split and a transform exists
        if self.apply_augmentation and self.augmentation_transform is not None:
            try:
                images_tensor = self.augmentation_transform(images_tensor)
            except Exception as e:
                # Fail soft: don't break data loading if aug fails for an edge case
                print(f"\u26a0 Augmentation failed for index {idx}: {e}")
        labels_tensor = torch.from_numpy(labels).float()
        return images_tensor, labels_tensor

    def create_augmentation_transform(self):
        """Create a torchvision.transforms pipeline from self.config."""
        aug_dict = self.config.augmentations
        if aug_dict is None:
            return None

        # Registry of allowed transforms
        registry = {
            "RandomHorizontalFlip": v2.RandomHorizontalFlip,
            "RandomVerticalFlip": v2.RandomVerticalFlip,
            "RandomRotation": v2.RandomRotation,
            "RandomAffine": v2.RandomAffine,
            "GaussianBlur": v2.GaussianBlur,
            "RandomErasing": v2.RandomErasing,
            "CenterCrop": v2.CenterCrop,
        }

        transforms_list = []

        for name, params in aug_dict.items():
            cls = registry.get(name)
            if cls is None:
                raise ValueError(f"Unknown torchvision transform name: {name}")
            transform = cls(**params)
            transforms_list.append(transform)

        return v2.Compose(transforms_list) if transforms_list else None


class ModularCellImageFeatureDataset(ModularCellImageDataset):
    """
    Dataset that returns selected images, transformed features, and fucci labels for each cell.
    """

    def __init__(self, data_config: DataSetConfig):
        if isinstance(data_config, dict):
            data_config = DataSetConfig(**data_config)

        super().__init__(data_config=data_config)
        self.feature_extractor = FeatureExtractor()
        self.feature_dataset = ModularCellFeaturesDataset(data_config=data_config)
        self.feature_transform = None
        self.fit_feature_transform()

    def __getitem__(self, idx: int):
        images_tensor, labels_tensor = super().__getitem__(idx)
        cell_data, labels = self.get_cell_data_fucci_labels(idx)
        features = self.feature_extractor.extract_all_features(cell_data)
        if self.feature_transform:
            transformed_feature = self.feature_transform.transform(
                pd.DataFrame(features, index=[0])
            )
        else:
            transformed_feature = list(features.values())
        features_arr = np.array(transformed_feature, dtype=np.float32)
        features_tensor = torch.from_numpy(features_arr)

        return (images_tensor, features_tensor), labels_tensor

    def fit_feature_transform(self):
        self.feature_transform = self.feature_dataset.feature_transform
        self.feature_dataset.feature_transform = None
        feature_df = self.feature_dataset.get_dataset_df()
        feature_array = feature_df.drop(columns=["label_488", "label_561"])
        self.feature_transform.fit(feature_array)


class ModularCellFeaturesDataset(BaseCellDataset):
    """
    Generic cell features dataset that works with any DataSource and Transform pipeline.
    """

    def __init__(
        self,
        data_config: DataSetConfig,
        use_cache: bool = True,
    ):
        if isinstance(data_config, dict):
            data_config = DataSetConfig(**data_config)

        self.use_cache = use_cache
        self.feature_extractor = FeatureExtractor()
        super().__init__(data_config=data_config)

        self.feature_transform = create_transform_pipeline_from_config(
            data_config.feature_transforms, transform_type="feature"
        )

        self.label_names = ["label_488", "label_561"]
        self.df = None

    def _get_cache_key(self) -> str:
        """Generate a cache key based on dataset configuration."""
        cache_params = {
            "cell_ids": sorted(self.cell_ids),
            "mask_intensity": self.mask_intensity,
            "image_transform_config": (
                self.image_transform.get_config() if self.image_transform else None
            ),
            "label_names": self.label_names,
        }
        cache_params.update(self.config.fucci_transform)
        cache_hash = make_hash_from_dict(cache_params, length=12)
        return f"features_{cache_hash}"

    def _get_cache_dir(self) -> Path:
        """Get the cache directory for storing feature files."""
        if hasattr(self.data_source, "path"):
            source_path = Path(self.data_source.path)
        elif hasattr(self.data_source, "data_source") and hasattr(
            self.data_source.data_source, "path"
        ):
            source_path = Path(self.data_source.data_source.path)
        else:
            raise ValueError("Cannot determine data source path for caching")
        cache_dir = source_path.parent / "features"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _get_cache_path(self) -> Path:
        cache_dir = self._get_cache_dir()
        cache_key = self._get_cache_key()
        return cache_dir / f"{cache_key}.parquet"

    def _load_cached_df(self) -> Optional[pd.DataFrame]:
        if not self.use_cache:
            return None
        cache_path = self._get_cache_path()
        df = load_parquet_cache(str(cache_path))
        return df

    def _save_cached_df(self, df: pd.DataFrame):
        if not self.use_cache:
            return
        cache_path = self._get_cache_path()
        save_parquet_cache(df, str(cache_path))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cell_data, labels = self.get_cell_data_fucci_labels(idx)
        features = self.feature_extractor.extract_all_features(cell_data)
        return features, labels

    def get_dataset_df(self, force_recompute: bool = False) -> pd.DataFrame:
        """Get dataset as a Pandas DataFrame with caching support.

        Args:
            force_recompute: If True, ignore cache and recompute features

        Returns:
            DataFrame with features and labels
        """
        df = None
        # Try to load from cache first
        if not force_recompute:
            cached_df = self._load_cached_df()
            if cached_df is not None:
                df = cached_df
        if df is None:
            # Cache miss or force recompute - extract features
            print(f"Extracting features from {len(self)} samples...")
            features_list = []
            labels_list = []
            for idx in tqdm(range(len(self)), desc="Extracting features"):
                try:
                    features, labels = self[idx]
                    features_list.append(features)
                    labels_list.append(labels)
                except Exception as e:
                    print(f"Warning: Failed to extract features for sample {idx}: {e}")
                    traceback.print_exc()
                    continue

            # Create DataFrame
            df = pd.DataFrame(features_list)
            for i, label_name in enumerate(self.label_names):
                df[label_name] = [label[i] for label in labels_list]

            # Save to cache
            self._save_cached_df(df)
        # # Remove outliers based on label_488 and label_561
        # for label in ["label_488", "label_561"]:
        #     q_low = df[label].quantile(0.01)
        #     q_high = df[label].quantile(0.99)
        #     df = df[(df[label] >= q_low) & (df[label] <= q_high)]
        #     print(f"Removed outliers in {label}: kept {len(df)} samples")

        if self.feature_transform:
            feature_df = df.drop(columns=self.label_names)
            print("Applying feature transformations...")
            self.feature_transform.fit(feature_df)
            feature_array = self.feature_transform.transform(feature_df)
            transformed_feature_df = pd.DataFrame(feature_array, columns=feature_df.columns)
            for i, label_name in enumerate(self.label_names):
                transformed_feature_df[label_name] = df[label_name].values
            df = transformed_feature_df

        self.df = df
        return self.df

    def split_X_y(self, dataset_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split the dataset into features (X) and labels (y).
        Returns:
            Tuple of (X, y) where:
            - X: Numpy array of features
            - y: Numpy array of labels with shape (N, 2) for [488_intensity, 561_intensity]
        """
        X = dataset_df.drop(columns=self.label_names).values
        y = dataset_df[self.label_names].values
        return X, y

    def split_set(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split the dataset into training, validation and testing subsets."""
        dataset_df = self.get_dataset_df()

        train_dataset, val_dataset, test_dataset = split_dataset(
            dataset=self,
            data_split=self.config.data_split,
            seed=self.config.seed,
        )
        # Map subset indices back into the DataFrame
        train_df = dataset_df.iloc[train_dataset.indices]
        val_df = dataset_df.iloc[val_dataset.indices]
        test_df = dataset_df.iloc[test_dataset.indices]

        return train_df, val_df, test_df


def split_dataset(
    dataset: Dataset, data_split: List[float], seed: int
) -> Tuple[Dataset, Dataset, Dataset]:
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset=dataset,
        lengths=data_split,
        generator=generator,
    )
    return train_dataset, val_dataset, test_dataset


def compute_fucci_labels(
    cell_data: CellData,
    mask_intensity: str = "segmentation",
    debug: bool = False,
) -> np.ndarray:
    """
    Compute FUCCI mean log intensities (488 and 561) from cell data within the specified mask.

    Returns:
        Array of shape (2,) with [intensity_488, intensity_561]
    """
    labels = []
    if mask_intensity not in ["segmentation", "nuclei_segmentation"]:
        raise ValueError(
            f"mask_intensity must be 'segmentation' or 'nuclei_segmentation' got {mask_intensity}"
        )
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
                raise ValueError(f"Missing mask '{mask_intensity}' for intensity computation")
            if len(planes) != len(masks):
                raise ValueError(f"Channel {channel} planes and segmentation planes count mismatch")
            # Compute mean intensity only within mask for each plane
            mask = masks[len(masks) // 2]  # Middle plane mask
            plane = planes[len(masks) // 2]  # Middle plane
            mean_outside_mask = plane[mask == 0].mean()
            plane_normalized = plane - mean_outside_mask  # Background subtraction
            plane_normalized = np.clip(plane_normalized, 0, None)  # Remove negatives
            if debug:
                show_images(
                    [plane, mask, plane_normalized],
                    titles=[f"{channel} plane", "Mask", "Normalized"],
                )
            masked_values = plane_normalized[mask > 0]
            mean_intensity = None
            if len(masked_values) > 0:
                mean_intensity = masked_values.mean()
                if np.isnan(mean_intensity):
                    raise ValueError(
                        f"NaN mean intensity for channel {channel} in cell {cell_data.metadata.get('cell_id', None)}"
                    )
            else:
                raise ValueError(f"No masked pixels found for channel {channel}")

            if mean_intensity is None:
                raise ValueError(f"No valid intensities found for channel {channel}")
            assert mean_intensity <= max(
                [plane.max() for plane in planes]
            ), "Mean intensity exceeds max plane intensity"
            labels.append(mean_intensity)
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
