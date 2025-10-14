"""
Iris ML Pipeline - Datenverarbeitung
Dieses Modul behandelt das Laden und Vorverarbeiten der Iris-Daten.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import os
import h5py
import json
import hashlib
from skimage.measure import regionprops, label
from skimage.io import imread
from tqdm import tqdm
from scipy import ndimage
from typing import Optional, Union, Dict, List, Tuple, Generator, Any
from abc import ABC, abstractmethod
from functools import lru_cache, partial

# PyTorch imports
from torch.utils.data import Dataset, DataLoader
import lightning as L
from torch.utils.data import random_split
from datetime import datetime
import traceback
import importlib

import yaml


def resolve(name: str):
    module_name, attr_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


class H5DataCleaner:
    """
    Analyzes H5 dataset and creates quality metadata file with valid cells.

    Filtering criteria:
    1. Cells without exactly 3 planes per channel
    2. Cells with multiple objects in segmentation (multiple cells)
    3. Cells where segmentation failed (empty mask)
    4. Cells where nuclei_seg is 1.2x or larger than cell seg

    Results are cached in a metadata file that includes the filtering criteria.
    If the same analysis is run again with the same criteria, results are loaded from cache.
    """

    def __init__(
        self, h5_file_path: str, nuclei_threshold: float = 1.2, max_objects: int = 1
    ):
        """
        Initialize the cleaner.

        Args:
            h5_file_path: Path to H5 file
            nuclei_threshold: Threshold ratio for nuclei/cell area (default 1.2)
            max_objects: Maximum allowed objects in segmentation (default 1)
        """
        self.h5_file_path = h5_file_path
        self.nuclei_threshold = nuclei_threshold
        self.max_objects = max_objects

        self.relevant_channels = ["405", "488", "561", "bf"]

        # Statistics
        self.stats = {
            "total_cells": 0,
            "invalid_plane_count": 0,
            "multiple_cells": 0,
            "failed_segmentation": 0,
            "nuclei_too_large": 0,
            "valid_cells": 0,
        }

        # Detailed results
        self.invalid_cells = {
            "invalid_plane_count": [],
            "multiple_cells": [],
            "failed_segmentation": [],
            "nuclei_too_large": [],
        }
        self.valid_cells = []

    def _get_criteria_hash(self) -> str:
        """
        Generate a hash of the filtering criteria.
        Used to identify cached metadata files.
        """
        criteria_str = (
            f"planes3_maxobj{self.max_objects}_nucthresh{self.nuclei_threshold:.2f}"
        )
        return hashlib.md5(criteria_str.encode()).hexdigest()[:8]

    def _get_metadata_filename(self) -> str:
        """
        Generate filename for metadata based on criteria.
        Format: <h5_basename>_quality_<criteria_hash>.json
        """
        base_name = os.path.splitext(self.h5_file_path)[0]
        criteria_hash = self._get_criteria_hash()
        return f"{base_name}_quality_{criteria_hash}.json"

    def _find_existing_metadata(self) -> Optional[str]:
        """
        Check if a metadata file with matching criteria exists.

        Returns:
            Path to metadata file if found, None otherwise
        """
        metadata_path = self._get_metadata_filename()

        if not os.path.exists(metadata_path):
            return None

        # Validate that the metadata is for the same file version
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            # Check file hash to ensure H5 file hasn't changed
            stat = os.stat(self.h5_file_path)
            current_hash = hashlib.md5(
                f"{self.h5_file_path}_{stat.st_size}_{stat.st_mtime}".encode()
            ).hexdigest()

            if metadata.get("file_hash") == current_hash:
                # Check criteria match
                stored_criteria = metadata.get("filtering_criteria", {})
                if (
                    stored_criteria.get("nuclei_to_cell_threshold")
                    == self.nuclei_threshold
                    and stored_criteria.get("max_objects_in_segmentation")
                    == self.max_objects
                ):
                    return metadata_path

            print(f"⚠ Found metadata but it's outdated or has different criteria")
            return None

        except Exception as e:
            print(f"⚠ Error reading metadata file: {e}")
            return None

    def _load_cached_metadata(self, metadata_path: str) -> Dict[str, Any]:
        """Load and return cached metadata."""
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Update internal state from metadata
        self.valid_cells = metadata.get("valid_cells", [])
        self.invalid_cells = metadata.get("invalid_cells", {})
        self.stats = metadata.get("statistics", {})

        return metadata

    def check_plane_count(self, cell_group: h5py.Group) -> bool:
        """Check if cell has exactly 3 planes per channel."""
        for channel in self.relevant_channels:
            if channel not in cell_group:
                continue

            n_planes = len(list(cell_group[channel].keys()))
            if n_planes != 3:
                return False

        return True

    def check_multiple_cells(
        self, mask: np.ndarray, max_objects: int = 1
    ) -> Tuple[bool, int]:
        """Check if segmentation mask contains multiple objects."""
        if mask is None or mask.size == 0:
            return True, 0

        # Label connected components
        labeled_mask = label(mask)
        num_objects = labeled_mask.max()

        return num_objects <= max_objects, num_objects

    def check_segmentation_failed(self, mask: np.ndarray) -> bool:
        """Check if segmentation failed (empty mask)."""
        if mask is None:
            return True

        return np.sum(mask > 0) == 0

    def compare_segmentation_areas(
        self, cell_seg: np.ndarray, nuclei_seg: np.ndarray
    ) -> Tuple[bool, float, float]:
        """Compare areas of cell vs nuclei segmentation."""
        if cell_seg is None or nuclei_seg is None:
            return False, 0, 0

        cell_area = np.sum(cell_seg > 0)
        nuclei_area = np.sum(nuclei_seg > 0)

        # Check if nuclei is larger than threshold
        is_too_large = nuclei_area > (self.nuclei_threshold * cell_area)

        return is_too_large, nuclei_area, cell_area

    def validate_cell(self, cell_group: h5py.Group, cell_name: str) -> Tuple[bool, str]:
        """Validate a single cell against all criteria."""
        # 1. Check plane count
        if not self.check_plane_count(cell_group):
            return False, "invalid_plane_count"

        # Get middle plane for segmentation checks
        plane_names = sorted(list(cell_group[self.relevant_channels[0]].keys()))
        if len(plane_names) < 3:
            return False, "invalid_plane_count"

        middle_plane = plane_names[len(plane_names) // 2]

        # Load segmentation masks for middle plane
        cell_seg = None
        nuclei_seg = None

        if "seg" in cell_group and middle_plane in cell_group["seg"]:
            cell_seg = cell_group["seg"][middle_plane][()]

        if "nuclei_seg" in cell_group and middle_plane in cell_group["nuclei_seg"]:
            nuclei_seg = cell_group["nuclei_seg"][middle_plane][()]

        # 2. Check if segmentation failed
        if self.check_segmentation_failed(cell_seg):
            return False, "failed_segmentation"

        # 3. Check for multiple cells
        is_single_cell, num_objects = self.check_multiple_cells(cell_seg, max_objects=1)
        if not is_single_cell:
            return False, f"multiple_cells"

        # 4. Check if nuclei is too large compared to cell
        if nuclei_seg is not None and cell_seg is not None:
            nuclei_too_large, nuclei_area, cell_area = self.compare_segmentation_areas(
                cell_seg, nuclei_seg
            )
            if nuclei_too_large:
                return False, "nuclei_too_large"

        return True, "valid"

    def analyze(
        self, show_progress: bool = True, force_reanalyze: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze the dataset and categorize cells.
        Checks for cached metadata first, only re-analyzes if cache doesn't exist or force_reanalyze=True.

        Args:
            show_progress: Whether to show progress bar
            force_reanalyze: Force re-analysis even if cached metadata exists

        Returns:
            Dictionary with analysis results and metadata
        """
        # Check for cached metadata first
        if not force_reanalyze:
            cached_path = self._find_existing_metadata()
            if cached_path:
                print(f"✓ Found cached metadata: {cached_path}")
                metadata = self._load_cached_metadata(cached_path)
                self.print_statistics()
                return metadata

        print(f"Analyzing dataset: {self.h5_file_path}")

        with h5py.File(self.h5_file_path, "r") as f:
            all_cells = list(f.keys())
            self.stats["total_cells"] = len(all_cells)

            print(f"Total cells in dataset: {len(all_cells):,}")

            iterator = (
                tqdm(all_cells, desc="Analyzing cells", unit="cell")
                if show_progress
                else all_cells
            )

            for cell_name in iterator:
                if cell_name not in f:
                    continue

                cell_grp = f[cell_name]
                is_valid, reason = self.validate_cell(cell_grp, cell_name)

                if is_valid:
                    self.valid_cells.append(cell_name)
                    self.stats["valid_cells"] += 1
                else:
                    # Categorize invalid cells
                    if reason == "invalid_plane_count":
                        self.invalid_cells["invalid_plane_count"].append(cell_name)
                        self.stats["invalid_plane_count"] += 1
                    elif reason == "multiple_cells":
                        self.invalid_cells["multiple_cells"].append(cell_name)
                        self.stats["multiple_cells"] += 1
                    elif reason == "failed_segmentation":
                        self.invalid_cells["failed_segmentation"].append(cell_name)
                        self.stats["failed_segmentation"] += 1
                    elif reason == "nuclei_too_large":
                        self.invalid_cells["nuclei_too_large"].append(cell_name)
                        self.stats["nuclei_too_large"] += 1

        # Print statistics
        self.print_statistics()

        # Create metadata
        stat = os.stat(self.h5_file_path)
        file_hash = hashlib.md5(
            f"{self.h5_file_path}_{stat.st_size}_{stat.st_mtime}".encode()
        ).hexdigest()
        criteria_hash = self._get_criteria_hash()

        metadata = {
            "h5_file_path": self.h5_file_path,
            "file_hash": file_hash,
            "file_size_bytes": stat.st_size,
            "file_mtime": stat.st_mtime,
            "analysis_date": datetime.now().isoformat(),
            "criteria_hash": criteria_hash,
            "filtering_criteria": {
                "require_3_planes": True,
                "max_objects_in_segmentation": self.max_objects,
                "reject_empty_segmentation": True,
                "nuclei_to_cell_threshold": self.nuclei_threshold,
            },
            "statistics": self.stats,
            "valid_cells": self.valid_cells,
            "invalid_cells": self.invalid_cells,
        }

        # Auto-save metadata to cache file
        metadata_path = self._get_metadata_filename()
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"\n✓ Metadata saved to: {metadata_path}")

        return metadata

    def print_statistics(self):
        """Print analysis statistics."""
        print("\n" + "=" * 60)
        print("DATASET QUALITY ANALYSIS")
        print("=" * 60)
        print(f"Total cells:                {self.stats['total_cells']:,}")
        print(f"Valid cells:                {self.stats['valid_cells']:,}")
        print(f"\nInvalid cells by reason:")
        print(f"  - Invalid plane count:    {self.stats['invalid_plane_count']:,}")
        print(f"  - Multiple cells:         {self.stats['multiple_cells']:,}")
        print(f"  - Failed segmentation:    {self.stats['failed_segmentation']:,}")
        print(f"  - Nuclei too large:       {self.stats['nuclei_too_large']:,}")
        print(
            f"\nTotal invalid:              {self.stats['total_cells'] - self.stats['valid_cells']:,}"
        )
        print(
            f"Retention rate:             {100 * self.stats['valid_cells'] / max(self.stats['total_cells'], 1):.1f}%"
        )
        print("=" * 60)

    def save_metadata(self, output_path: str = None) -> str:
        """
        Save quality metadata to JSON file.

        Args:
            output_path: Path to save metadata (default: auto-generate)

        Returns:
            Path to saved metadata file
        """
        if output_path is None:
            # Auto-generate filename
            base_name = os.path.splitext(self.h5_file_path)[0]
            output_path = f"{base_name}_quality_metadata.json"

        metadata = self.analyze(show_progress=True)

        print(f"\nSaving metadata to: {output_path}")
        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Metadata saved successfully")
        return output_path


class H5CellDataset(Dataset):
    """
    PyTorch Dataset für H5-Zellbilddaten mit automatischer Qualitätsprüfung.

    Verwendet H5DataCleaner um ungültige Zellen zu filtern.
    Ergebnisse werden automatisch in einer Metadaten-Datei gecacht.

    Args:
        h5_file_path: Path to H5 file
        transform: Optional transform for input images (e.g., ImagePreprocessor)
        target_transform: Optional transform for output/targets (e.g., FUCCIRepresentationTransform)
        use_quality_filter: Enable quality filtering
        nuclei_threshold: Threshold for nuclei/cell size ratio
        max_objects: Max objects in segmentation
        force_reanalyze: Force re-analysis even if cache exists
    """

    def __init__(
        self,
        h5_file_path: str,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
        feature_extractor: Optional[Any] = None,
        use_quality_filter: bool = True,
        nuclei_threshold: float = 1.2,
        max_objects: int = 1,
        force_reanalyze: bool = False,
    ):
        """
        Args:
            h5_file_path: Pfad zur H5-Datei
            transform: Optional transform for input images (applied to raw images)
            target_transform: Optional transform for targets/labels (applied to outputs)
            nuclei_threshold: Threshold für Nuclei/Cell Größenverhältnis (default 1.2)
            max_objects: Max. erlaubte Objekte in Segmentierung (default 1)
            use_quality_filter: Wenn True, führe Qualitätsfilter durch (default True)
            force_reanalyze: Erzwinge Neuanalyse auch wenn Cache existiert (default False)
        """
        super().__init__()

        self.h5_file_path = h5_file_path
        self.transform = transform
        self.target_transform = target_transform
        self.feature_extractor = feature_extractor
        self.nuclei_threshold = nuclei_threshold
        self.max_objects = max_objects

        if not os.path.exists(h5_file_path):
            raise FileNotFoundError(f"H5-Datei nicht gefunden: {h5_file_path}")

        # Load cell names using quality filter or all cells
        if use_quality_filter:
            # Use H5DataCleaner to filter cells (with automatic caching)
            print(
                f"Using quality filter with nuclei_threshold={nuclei_threshold}, max_objects={max_objects}"
            )
            cleaner = H5DataCleaner(
                h5_file_path, nuclei_threshold=nuclei_threshold, max_objects=max_objects
            )
            metadata = cleaner.analyze(
                show_progress=True, force_reanalyze=force_reanalyze
            )
            self.cell_names = metadata["valid_cells"]
            print(f"✓ Loaded {len(self.cell_names):,} valid cells")
        else:
            # Load all cells without filtering
            print("Loading all cells without quality filter")
            with h5py.File(h5_file_path, "r") as f:
                self.cell_names = list(f.keys())
            print(f"✓ Loaded {len(self.cell_names):,} cells (no filtering)")

    def __len__(self) -> int:
        """Gibt die Anzahl der Zellen zurück."""
        return len(self.cell_names)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        Lädt eine einzelne Zelle.

        Args:
            idx: Index der Zelle

        Returns:
            Tuple (image, target) where transforms have been applied
        """
        if idx >= len(self.cell_names):
            raise IndexError(
                f"Index {idx} außerhalb des Bereichs [0, {len(self.cell_names)})]"
            )

        cell_name = self.cell_names[idx]

        # Load raw cell data
        cell_data = self._load_cell_data(cell_name)

        # Apply transform to input images
        result = {}
        if self.transform:
            result["images"] = self.transform(cell_data)

        # Apply target_transform to create target/label
        if self.target_transform:
            result["labels"] = self.target_transform(cell_data)

        if self.feature_extractor:
            result["features"] = self.feature_extractor(cell_data)

        return result

    def _load_cell_data(
        self, cell_name: str
    ) -> Dict[str, List[Tuple[str, np.ndarray]]]:
        """Lädt RAW Daten für eine einzelne Zelle (ohne Verarbeitung)."""
        with h5py.File(self.h5_file_path, "r") as f:
            if cell_name not in f:
                raise KeyError(f"Zelle {cell_name} nicht gefunden")

            grp = f[cell_name]
            cell_data = {"cell_name": cell_name}
            for channel_name, channel_data in grp.items():
                cell_data[channel_name] = []
                cell_data["planes"] = channel_data.keys()  # list of planes
                for plane_name, plane_data in channel_data.items():
                    image = plane_data[()]
                    cell_data[channel_name].append(image)

            return cell_data

    def get_features_dataframe(
        self, indices: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Extrahiert Features für alle oder ausgewählte Zellen als DataFrame.

        Args:
            indices: Liste von Indices (default: alle)

        Returns:
            DataFrame mit Features
        """
        if not self.feature_extractor:
            raise ValueError("Feature-Extraktion nicht aktiviert")

        if indices is None:
            indices = list(range(len(self)))

        features_list = []
        for idx in tqdm(indices, desc="Extrahiere Features"):
            item = self[idx]
            if "features" in item:
                features_list.append(item["features"])

        return pd.DataFrame(features_list)


class CellImageDataset(H5CellDataset):
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        channel_images = data.get("images", None)
        labels = data.get("labels", None)
        labels = list(labels.values())
        images = []
        images.extend(channel_images.get("bf", []))
        images.extend(channel_images.get("405", []))
        images = np.stack(images, axis=2)

        if images.shape != (250, 250, 6):
            raise ValueError(f"Unexpected image shape: {images.shape}")
        return images, labels


class CellImageTransform:
    """
    Transform class for processing input cell images (bf and 405 channels only).

    Applies preprocessing steps like:
    - Gaussian filtering
    - Illumination correction
    - Normalization
    - Plane selection

    Note: This transform only processes INPUT channels (bf, 405) for the CNN.
          Target channels (488, 561) are handled by CellTargetTransform.

    Usage:
        transform = CellImageTransform(
            plane_selection='middle',
            apply_gaussian=True
        )
        processed_images = transform(raw_cell_data)
    """

    def __init__(
        self,
        plane_selection: str = "all",
        transforms: Optional[List[Any]] = None,
    ):
        """
        Args:
            plane_selection: Which planes to select ('middle', 'first', 'last', 'all')
            apply_gaussian: Whether to apply Gaussian filtering
            sigma: Sigma for Gaussian filter
            apply_illumination_correction: Whether to correct illumination
            normalize: Whether to normalize images
        """
        # Hardcoded input channels for CNN
        self.channels = ["bf", "405"]
        self.plane_selection = plane_selection
        self.transforms = transforms if transforms is not None else []

    def __call__(self, cell_data: Dict) -> np.ndarray:
        """
        Process cell data and return stacked image array.

        Args:
            cell_data: Dictionary with channel data from H5 file

        Returns:
            Stacked numpy array of shape (height, width, n_channels * n_planes)
        """
        processed_images_channels = {channel: [] for channel in self.channels}

        for channel in self.channels:
            if channel not in cell_data:
                continue

            # Get plane data (list of tuples: [(plane_name, image), ...])
            images = cell_data[channel]

            # transform each image
            for image in images:
                for transform in self.transforms:
                    image = transform(image)
                processed_images_channels[channel].append(image)

        # Stack all processed images along channel dimension
        return {
            channel: images for channel, images in processed_images_channels.items()
        }


class FUCCIRepresentationTransform:
    """
    Transform class for computing FUCCI representation (TARGET channels: 488, 561).

    Applies the same preprocessing as CellImageTransform (Gaussian, illumination correction,
    normalization) to FUCCI channels and extracts intensity values as regression targets.

    Usage:
        target_transform = FUCCIRepresentationTransform(
            plane_selection='middle',
            apply_gaussian=True
        )
        fucci_target = target_transform(raw_cell_data)
    """

    def __init__(
        self,
        plane_selection: str = "all",
        apply_illumination_correction: bool = False,
        repr_function: str = "mean",
    ):
        """
        Args:
            plane_selection: Which planes to use ('middle', 'first', 'last', 'all')
            apply_gaussian: Whether to apply Gaussian filtering
            sigma: Sigma for Gaussian filter
            apply_illumination_correction: Whether to correct illumination
            normalize: Whether to normalize images
            repr_function: How to aggregate intensities ('mean', 'median', 'sum')
        """
        # Hardcoded target channels for regression
        self.channels = ["488", "561"]
        self.plane_selection = plane_selection
        self.apply_illumination_correction = apply_illumination_correction
        self.repr_function = repr_function

    def __call__(self, cell_data: Dict) -> Dict:
        """
        Compute FUCCI representation from cell data with preprocessing.

        Args:
            cell_data: Dictionary with channel data from H5 file

        Returns:
            Numpy array of shape (n_channels,) with processed intensity values
        """
        intensities = {}
        masks = cell_data.get("seg", None)

        for channel in self.channels:
            if channel not in cell_data:
                raise KeyError(f"Channel {channel} not found in cell data")

            if self.apply_illumination_correction:
                images = self._remove_illumination(cell_data[channel], masks)
            else:
                images = cell_data[channel]

            # Select planes
            intensities[channel] = self._compute_intensity(
                self._select_planes(images), self._select_planes(masks)
            )

        return intensities

    def _select_planes(self, planes: List[np.ndarray]) -> List[np.ndarray]:
        """Select planes based on selection strategy."""
        if not planes:
            raise ValueError("No planes available for selection")

        if self.plane_selection == "middle":
            mid_idx = len(planes) // 2
            return [planes[mid_idx]]
        elif self.plane_selection == "first":
            return [planes[0]]
        elif self.plane_selection == "last":
            return [planes[-1]]
        elif self.plane_selection == "all":
            return planes
        else:
            raise ValueError(f"Unknown plane_selection: {self.plane_selection}")

    def _compute_intensity(
        self, images: List[np.ndarray], masks: List[np.ndarray]
    ) -> float:
        """
        Compute representative intensity across multiple planes with optional masking.
        """

        if not masks:
            raise ValueError("Segmentation mask is required for intensity computation")
        all_masked_values = []
        for image, mask in zip(images, masks):
            if mask is not None and np.sum(mask > 0) > 0:
                masked_values = image[mask > 0]
                all_masked_values.extend(masked_values)

        if not all_masked_values:
            raise ValueError("No valid masked pixels found across all planes")

        if self.repr_function == "mean":
            return float(np.mean(all_masked_values))
        elif self.repr_function == "median":
            return float(np.median(all_masked_values))
        elif self.repr_function == "sum":
            return float(np.sum(all_masked_values))
        else:
            raise ValueError(f"Unknown repr_function: {self.repr_function}")

    def _remove_illumination(
        self, images: List[np.ndarray], masks: List[np.ndarray]
    ) -> np.ndarray:
        """Remove illumination effects based on mask."""
        corrected = []
        for mask, image in zip(masks, images):
            if mask is None or mask.sum() == 0:
                raise ValueError(
                    "Segmentation mask is required for illumination correction"
                )

            mean_intensity = np.median(image[mask == 0])
            corrected = image - mean_intensity
            corrected[corrected < 0] = 0
            corrected.append(corrected)

        return corrected


class FeatureExtractor:
    """Klasse für Feature-Extraktion - getrennt von der Datenladung."""

    def extract_morphological_features(
        self, image: np.ndarray, mask: np.ndarray, cell_name: str, type: str
    ) -> Dict:
        """Extrahiert morphologische Features aus Zellbildern."""
        labeled_mask = label(mask)
        properties = regionprops(labeled_mask, intensity_image=image)

        if not properties:
            return self._get_empty_morphological_features(cell_name, type)

        prop = properties[0]
        return {
            f"{type}_area": prop.area,
            f"{type}_perimeter": prop.perimeter,
            f"{type}_mean_intensity": prop.mean_intensity,
            f"{type}_eccentricity": prop.eccentricity,
            f"{type}_solidity": prop.solidity,
            f"{type}_extent": prop.extent,
            f"{type}_major_axis_length": prop.major_axis_length,
            f"{type}_minor_axis_length": prop.minor_axis_length,
        }

    def extract_intensity_features(
        self, image: np.ndarray, mask: np.ndarray, channel: str
    ) -> Dict:
        """Extrahiert Intensitäts-Features."""
        # Intensität innerhalb und außerhalb der Maske
        total_intensity_inside = np.sum(image[mask > 0])
        total_intensity_outside = np.sum(image[mask == 0])
        total_intensity = total_intensity_inside + total_intensity_outside

        features = {
            "intensity_inside": total_intensity_inside,
            "intensity_outside": total_intensity_outside,
        }

        if total_intensity > 0:
            features["intensity_ratio_outside_inside"] = (
                total_intensity_outside / total_intensity_inside
                if total_intensity_inside > 0
                else np.inf
            )
            features["intensity_fraction_outside"] = (
                total_intensity_outside / total_intensity
            )
        else:
            features["intensity_ratio_outside_inside"] = 0
            features["intensity_fraction_outside"] = 0

        return features

    def extract_channel_statistics(
        self, image: np.ndarray, mask: np.ndarray, channel: str
    ) -> Dict:
        """Extrahiert Kanalstatistiken."""
        masked_pixels = image[mask > 0]
        stats = {}

        if masked_pixels.size > 0:
            stats[f"mean_intensity_in_mask_{channel}"] = np.mean(masked_pixels)
            stats[f"median_intensity_in_mask_{channel}"] = np.median(masked_pixels)
            stats[f"std_intensity_in_mask_{channel}"] = np.std(masked_pixels)
            stats[f"min_intensity_in_mask_{channel}"] = np.min(masked_pixels)
            stats[f"max_intensity_in_mask_{channel}"] = np.max(masked_pixels)
        else:
            stats[f"mean_intensity_in_mask_{channel}"] = 0
            stats[f"median_intensity_in_mask_{channel}"] = 0
            stats[f"std_intensity_in_mask_{channel}"] = 0
            stats[f"min_intensity_in_mask_{channel}"] = 0
            stats[f"max_intensity_in_mask_{channel}"] = 0

        return stats

    def extract_all_features(self, cell_data: Dict, cell_name: str) -> Dict:
        """Extrahiert alle Features für eine Zelle."""
        features = {"cell_name": cell_name}

        # Prüfe ob alle erforderlichen Kanäle vorhanden sind
        required_channels = ["405", "seg", "561", "488"]
        if not all(ch in cell_data for ch in required_channels):
            return features

        try:
            # Nimm mittlere Ebene jedes Kanals
            middle_idx = len(cell_data["405"]) // 2

            nucleus_image = cell_data["405"][middle_idx]
            seg_mask = cell_data["seg"][middle_idx]
            red_image = cell_data["561"][middle_idx]
            green_image = cell_data["488"][middle_idx]
            nuclei_seg_mask = cell_data["nuclei_seg"][middle_idx]
            # Morphologische Features
            morphological_features_cell = self.extract_morphological_features(
                nucleus_image, seg_mask, cell_name, type="cell"
            )
            if "nuclei_seg" in cell_data:
                morphological_features_nucleus = self.extract_morphological_features(
                    nucleus_image, nuclei_seg_mask, cell_name, type="nucleus"
                )

            features.update(morphological_features_cell)
            features.update(morphological_features_nucleus)

            features["cell_nucleus_area_ratio"] = (
                morphological_features_nucleus["nucleus_area"]
                / morphological_features_cell["cell_area"]
                if morphological_features_cell["cell_area"] > 0
                else 0
            )
            # Intensitäts-Features
            intensity_features = self.extract_intensity_features(
                nucleus_image, seg_mask, "405"
            )
            features.update(intensity_features)

            # Kanalstatistiken
            red_stats = self.extract_channel_statistics(red_image, seg_mask, "561")
            green_stats = self.extract_channel_statistics(green_image, seg_mask, "488")
            features.update(red_stats)
            features.update(green_stats)

        except Exception as e:
            print(f"Fehler beim Extrahieren der Features für {cell_name}:")
            traceback.print_exc()

        return features

    def _get_empty_morphological_features(self, cell_name: str, type: str) -> Dict:
        """Gibt leere morphologische Features zurück."""
        return {
            f"{type}_area": 0,
            f"{type}_perimeter": 0,
            f"{type}_mean_intensity": 0,
            f"{type}_eccentricity": 0,
            f"{type}_solidity": 0,
            f"{type}_extent": 0,
            f"{type}_major_axis_length": 0,
            f"{type}_minor_axis_length": 0,
        }


class CellDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        data_config_path: str = "configs/data_config.yaml",
    ):

        super().__init__()
        with open(data_config_path, "r") as f:
            config = yaml.safe_load(f)
        self.save_hyperparameters()
        self.dataset_config = config.get("data_set_config", {})
        self.transforms_config = config.get("transforms_config", {})
        self.transforms_target = config.get("transforms_target", {})
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_validation_split = config.get("train_val_split", 0.8)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Teilt das Dataset in Trainings-, Validierungs- und Testdaten auf."""
        transform_config_list = self.transforms_config.pop("transforms", [])
        transform_list = [
            partial(resolve(cfg["class_path"]), **cfg["init_args"])
            for cfg in transform_config_list
        ]
        self.image_transform = CellImageTransform(
            **self.transforms_config, transforms=transform_list
        )
        self.target_transform = FUCCIRepresentationTransform(**self.transforms_target)

        self.dataset = CellImageDataset(
            **self.dataset_config,
            transform=self.image_transform,
            target_transform=self.target_transform,
        )
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset,
            [
                self.train_validation_split,
                (1 - self.train_validation_split) / 2,
                (1 - self.train_validation_split) / 2,
            ],
        )

    def train_dataloader(self):
        """Gibt den DataLoader für das Training zurück."""
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        """Gibt den DataLoader für die Validierung zurück."""
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        """Gibt den DataLoader für den Test zurück."""
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )


if __name__ == "__main__":
    from scipy.ndimage import gaussian_filter

    # Beispielhafte Nutzung
    h5_path = (
        "/myhome/iris/data/fucci_3t3_221124_filtered_noNG030JP208_with_nuclei_seg.h5"
    )
    preprocessor = CellImageTransform(
        transforms=[lambda x: gaussian_filter(x, sigma=1.0)]
    )
    feature_extractor = FeatureExtractor()

    dataset = CellImageDataset(
        h5_file_path=h5_path,
        transform=preprocessor,
        target_transform=FUCCIRepresentationTransform(
            plane_selection="all", repr_function="mean"
        ),
    )
    print(f"Dataset Größe: {len(dataset)} Zellen")
    print("Beispiel Zelle laden...")
    example_cell = dataset[0]
    print(f"✓ Geladene Beispielzelle: {example_cell}")
