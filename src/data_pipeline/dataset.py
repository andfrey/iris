"""
Modern PyTorch Dataset implementation using the new modular architecture.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import numpy as np
import lightning as L
from tqdm import tqdm
import yaml
from .utils import (
    save_parquet_cache,
    load_parquet_cache,
    make_hash_from_dict,
    create_data_source_from_config,
    create_transform_pipeline_from_config,
)
import pandas as pd


from .data_sources import DataSource, CellData
from .data_transforms import Transform
from .feature_extractor import FeatureExtractor


# === BaseCellDataset ===
class BaseCellDataset(Dataset):
    def __init__(
        self,
        data_source: DataSource,
        transform: Optional[Transform] = None,
        mask_intensity: str = "segmentation",
        debug: bool = False,
        scale_divider_488: float = 1.0,
        scale_divider_561: float = 1.0,
    ):
        self.data_source = data_source
        self.transform = transform
        self.mask_intensity = mask_intensity
        self.debug = debug
        self.scale_divider_488 = scale_divider_488
        self.scale_divider_561 = scale_divider_561
        print(f"✓ Scale dividers: 488={self.scale_divider_488}, 561={self.scale_divider_561}")
        self.fucci_scaler = np.array([self.scale_divider_488, self.scale_divider_561])
        self.cell_ids = data_source.get_cell_ids()
        print(f"✓ Dataset initialized with {len(self.cell_ids):,} cells")

    def __len__(self) -> int:
        return len(self.cell_ids)

    def get_cell_data(self, idx: int):
        cell_id = self.cell_ids[idx]
        cell_data = self.data_source.load_cell(cell_id)
        if self.transform:
            cell_data = self.transform(cell_data)
        return cell_data

    def compute_labels(self, cell_data):
        labels = compute_fucci_labels(
            cell_data, mask_intensity=self.mask_intensity, debug=self.debug
        )
        labels = labels / self.fucci_scaler
        return labels


class ModularCellDataModule(L.LightningDataModule):
    """
    Lightning DataModule for the cell dataset.

    Using the ModularCellDataset assembling configurable DataSource, filters, and transforms into
    a Pytorch Dataset subclass.
    Which is used to create dataloaders for training, validation, testing, and prediction.

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
            - class_path: src.data_pipeline.data_transforms.NormalizeTransform
            - init_args:
                method: standardize  # Changed to standardize for zero mean and unit variance
                channel_keys: ['405', 'bf']
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

                            Transform parameters (ordered list of transforms):
                            transforms:
                            - class_path: src.data_pipeline.data_transforms.NormalizeTransform
                            init_args:
                                method: standardize  # Changed to standardize for zero mean and unit variance
                                channel_keys: ['405', 'bf']
        """
        super().__init__()
        with open(data_config_path, "r") as f:
            self.config = yaml.safe_load(f)
        # Save hyperparameters for Lightning
        self.save_hyperparameters(self.config)

        # Store parameters
        self.h5_path = self.config.get("h5_path", None)
        # Resolve relative path
        if not Path(self.h5_path).exists():
            abs_path = Path(__file__).resolve().parent.parent / self.h5_path
            if Path(abs_path).exists():
                self.h5_path = abs_path
            else:
                raise FileNotFoundError(f"H5 file not found: {self.h5_path}")

        self.batch_size = self.config.get("batch_size", 64)
        self.num_workers = self.config.get("num_workers", 0)
        self.data_split = self.config.get("data_split", [0.7, 0.2, 0.1])
        self.seed = self.config.get("seed", 42)  # For reproducibility

        if self.config.get("fucci_scale_transform", None) is not None:
            if (
                not "scale_divider_488" in self.config["fucci_scale_transform"]
                or not "scale_divider_561" in self.config["fucci_scale_transform"]
            ):
                raise ValueError(
                    "fucci_scale_transform requires 'scale_divider_488' and 'scale_divider_561' in data_config"
                )
            self.scale_divider_488 = self.config["fucci_scale_transform"]["scale_divider_488"]
            self.scale_divider_561 = self.config["fucci_scale_transform"]["scale_divider_561"]
        else:
            self.scale_divider_488 = 1.0
            self.scale_divider_561 = 1.0

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

        data_source = create_data_source_from_config(self.config)
        transform_pipeline = create_transform_pipeline_from_config(
            self.config, transform_type="image"
        )

        # 4. Create full dataset
        print(f"\n4. Creating PyTorch dataset")
        self.full_dataset = ModularCellImageDataset(
            data_source=data_source,
            transform=transform_pipeline,
            scale_divider_488=self.scale_divider_488,
            scale_divider_561=self.scale_divider_561,
        )

        # 5. Split into train/val
        if stage == "fit" or stage is None:
            print(f"\n5. Splitting dataset: {self.data_split}")

            # Split with generator for reproducibility
            generator = torch.Generator().manual_seed(self.seed)
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                dataset=self.full_dataset,
                lengths=self.data_split,
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
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.num_workers > 0 else False,
        )

    def predict_dataloader(self) -> DataLoader:
        """Returns prediction dataloader (full dataset)."""
        return DataLoader(
            self.full_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.num_workers > 0 else False,
        )


class ModularCellImageDataset(BaseCellDataset):
    """
    Generic cell image dataset that works with any DataSource and Transform pipeline.
    """

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cell_data = self.get_cell_data(idx)
        labels = self.compute_labels(cell_data)
        channel_list = []
        for channel_name in ["bf", "405"]:
            for plane in cell_data.channels.get(channel_name, None):
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
        labels_tensor = torch.from_numpy(labels).float()
        return images_tensor, labels_tensor


class ModularCellFeaturesDataset(BaseCellDataset):
    """
    Generic cell features dataset that works with any DataSource and Transform pipeline.
    """

    def __init__(
        self,
        data_config: dict,
        mask_intensity: str = "segmentation",
        debug: bool = False,
        use_cache: bool = True,
    ):
        # Set up transforms and config
        self.config = data_config
        self.use_cache = use_cache
        self.image_transform = create_transform_pipeline_from_config(
            self.config, transform_type="image"
        )
        self.feature_transform = create_transform_pipeline_from_config(
            self.config, transform_type="feature"
        )
        # Create data source
        data_source = create_data_source_from_config(self.config)
        # Call base class init
        super().__init__(
            data_source=data_source,
            transform=self.image_transform,
            mask_intensity=mask_intensity,
            debug=debug,
            scale_divider_488=self.config.get("fucci_scale_transform", {}).get(
                "scale_divider_488", 1.0
            ),
            scale_divider_561=self.config.get("fucci_scale_transform", {}).get(
                "scale_divider_561", 1.0
            ),
        )
        self.feature_extractor = FeatureExtractor()
        self.df = None

    def _get_cache_key(self) -> str:
        """Generate a cache key based on dataset configuration."""
        cache_params = {
            "cell_ids": sorted(self.cell_ids),
            "mask_intensity": self.mask_intensity,
            "image_transform_config": (
                self.image_transform.get_config() if self.image_transform else None
            ),
            "fucci_scale_transform": {
                "scale_divider_488": self.scale_divider_488,
                "scale_divider_561": self.scale_divider_561,
            },
        }
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
        cell_data = self.get_cell_data(idx)
        labels = self.compute_labels(cell_data)
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
                    continue

            # Create DataFrame
            df = pd.DataFrame(features_list)
            df["label_488"] = [label[0] for label in labels_list]
            df["label_561"] = [label[1] for label in labels_list]

            # Save to cache
            self._save_cached_df(df)
        # # Remove outliers based on label_488 and label_561
        # for label in ["label_488", "label_561"]:
        #     q_low = df[label].quantile(0.01)
        #     q_high = df[label].quantile(0.99)
        #     df = df[(df[label] >= q_low) & (df[label] <= q_high)]
        #     print(f"Removed outliers in {label}: kept {len(df)} samples")

        if self.feature_transform:
            feature_df = df.drop(columns=["label_488", "label_561"])
            print("Applying feature transformations...")
            self.feature_transform.fit(feature_df)
            feature_array = self.feature_transform.transform(feature_df)
            transformed_feature_df = pd.DataFrame(feature_array, columns=feature_df.columns)
            transformed_feature_df["label_488"] = df["label_488"].values
            transformed_feature_df["label_561"] = df["label_561"].values
            df = transformed_feature_df

        self.df = df
        return self.df

    @staticmethod
    def split_X_y(dataset_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split the dataset into features (X) and labels (y).
        Returns:
            Tuple of (X, y) where:
            - X: Numpy array of features
            - y: Numpy array of labels with shape (N, 2) for [488_intensity, 561_intensity]
        """
        X = dataset_df.drop(columns=["label_488", "label_561"]).values
        y = dataset_df[["label_488", "label_561"]].values
        return X, y

    def split_train_test_set(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the dataset into training and testing subsets.

        Args:
            train_ratio: Proportion of data to use for training (default: 0.8)
            seed: Random seed for reproducibility

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        # Set random seed for reproducibility
        np.random.seed(self.config.get("seed", 42))
        train_ratio = self.config.get("train_test_ratio", 0.9)

        dataset_df = self.get_dataset_df()
        # Shuffle dataset indices
        indices = np.random.permutation(len(dataset_df))
        train_size = int(len(dataset_df) * train_ratio)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        train_dataset = dataset_df.iloc[train_indices]
        test_dataset = dataset_df.iloc[test_indices]

        return train_dataset, test_dataset


def compute_fucci_labels(
    cell_data: CellData,
    mask_intensity: str = "segmentation",
    log_transform: bool = False,
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
            intensities = []
            for plane, mask in zip(planes, masks):
                mean_outside_mask = plane[mask == 0].mean()
                plane_normalized = plane - mean_outside_mask  # Background subtraction
                plane_normalized = np.clip(plane_normalized, 0, None)  # Remove negatives
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
            if log_transform:
                mean_intensity = (
                    np.log(mean_intensity) if mean_intensity > 0.0 else 0.0
                )  # Log normalization

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
