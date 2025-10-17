"""
Data source adapters for different storage formats.
Each adapter provides a unified interface for reading data.
"""

import json
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
import h5py
import numpy as np
from tqdm import tqdm
import hashlib
import os

from .data_filters import CellFilter, CompositeFilter, FilterStatistics, FilterResult


@dataclass
class CellData:
    """Standardized cell data structure"""

    cell_id: str
    channels: Dict[
        str, List[np.ndarray]
    ]  # e.g., {'405': [array1, array2], '488': [array1, array2]}
    metadata: Dict[str, Any] = field(default_factory=dict)
    segmentation: Optional[np.ndarray] = None
    nuclei_segmentation: Optional[np.ndarray] = None


class DataSource(ABC):
    """Abstract base class for data sources"""

    def __init__(self, path: str):
        self.path = Path(path)
        self._validate_path()

    def _validate_path(self):
        """Validate that the data source exists"""
        if not self.path.exists():
            raise FileNotFoundError(f"Data source not found: {self.path}")

    @abstractmethod
    def get_cell_ids(self) -> List[str]:
        """Return list of all cell IDs"""
        pass

    @abstractmethod
    def load_cell(self, cell_id: str) -> CellData:
        """Load a single cell's data"""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset-level metadata"""
        pass

    def __len__(self) -> int:
        return len(self.get_cell_ids())


class H5DataSource(DataSource):
    """Data source for HDF5 files with cell microscopy data"""

    def __init__(
        self,
        path: str,
        channel_keys: List[str] = ["405", "488", "561", "bf"],
        seg_key: str = "seg",
        nuclei_seg_key: str = "nuclei_seg",
        plane_selection: str = "all",  # 'all', 'middle', 'first', 'last'
    ):
        super().__init__(path)
        self.channel_keys = channel_keys
        self.seg_key = seg_key
        self.nuclei_seg_key = nuclei_seg_key
        self.plane_selection = plane_selection
        self._cell_ids = None

    def get_cell_ids(self) -> List[str]:
        """Get all cell IDs from H5 file"""
        if self._cell_ids is None:
            with h5py.File(self.path, "r") as f:
                # Filter out non-cell keys (e.g., metadata)
                self._cell_ids = [key for key in f.keys() if key.isdigit()]
        return self._cell_ids

    def load_cell(self, cell_id: str) -> CellData:
        """Load cell data from H5 file"""
        with h5py.File(self.path, "r") as f:
            if cell_id not in f:
                raise KeyError(f"Cell {cell_id} not found in {self.path}")

            cell_group = f[cell_id]

            # Load channels
            channels = {}
            for channel_key in self.channel_keys:
                if channel_key in cell_group:
                    planes = self._load_planes(cell_group[channel_key])
                    channels[channel_key] = planes

            # Load segmentation
            segmentation = None
            if self.seg_key in cell_group:
                segmentation = self._load_planes(cell_group[self.seg_key])

            # Load nuclei segmentation
            nuclei_seg = None
            if self.nuclei_seg_key in cell_group:
                nuclei_seg = self._load_planes(cell_group[self.nuclei_seg_key])

            # Extract metadata from attributes
            metadata = dict(cell_group.attrs)
            metadata["cell_id"] = cell_id

            return CellData(
                cell_id=cell_id,
                channels=channels,
                metadata=metadata,
                segmentation=segmentation,
                nuclei_segmentation=nuclei_seg,
            )

    def _load_planes(self, group) -> List[np.ndarray]:
        """
        Load planes from H5 group.

        Returns:
            List of 2D arrays (one per plane)
        """
        # Check if it's a group (with planes) or a dataset
        if isinstance(group, h5py.Dataset):
            # It's a single dataset (old format or already processed)
            return [group[()]]

        # It's a group with multiple planes
        plane_keys = group.keys()

        if self.plane_selection == "all":
            # Return all planes
            return [group[key][()] for key in plane_keys]

        elif self.plane_selection == "middle":
            idx = len(plane_keys) // 2
            return [group[plane_keys[idx]][()]]

        elif self.plane_selection == "first":
            return [group[plane_keys[0]][()]]

        elif self.plane_selection == "last":
            return [group[plane_keys[-1]][()]]

        else:
            raise ValueError(f"Unknown plane_selection: {self.plane_selection}")

    def get_metadata(self) -> Dict[str, Any]:
        """Get file-level metadata"""
        with h5py.File(self.path, "r") as f:
            metadata = dict(f.attrs) if f.attrs else {}
            metadata["num_cells"] = len(self.get_cell_ids())
            metadata["file_path"] = str(self.path)
            return metadata


"""
Filtered dataset that applies quality filters to data sources.
"""


class FilteredDataSource:
    """
    Wrapper around DataSource that filters cells based on quality criteria.

    This class applies filters and caches the list of valid cell IDs.
    Results are cached to avoid re-filtering on subsequent runs.
    """

    def __init__(
        self,
        data_source: DataSource,
        filters: List[CellFilter],
        cache_results: bool = True,
        force_refilter: bool = False,
    ):
        """
        Args:
            data_source: Underlying data source
            filters: List of filters to apply
            cache_results: Whether to cache filter results to file
            force_refilter: Force re-filtering even if cache exists
        """
        self.data_source = data_source
        self.composite_filter = CompositeFilter(filters)
        self.cache_results = cache_results
        self.force_refilter = force_refilter

        # Lazy evaluation - only filter when needed
        self._valid_cell_ids: Optional[List[str]] = None
        self._filter_stats: Optional[FilterStatistics] = None

        # Try to load from cache first
        if cache_results and not force_refilter:
            self._load_from_cache()

    def _get_cache_filename(self) -> str:
        """Generate cache filename based on data source and filters"""
        # Create a hash of the filter configuration
        filter_names = "_".join(self.composite_filter.get_filter_names())
        filter_hash = hashlib.md5(filter_names.encode()).hexdigest()[:8]

        # Get base name from data source path
        base_name = os.path.splitext(str(self.data_source.path))[0]
        return f"{base_name}_filtered_{filter_hash}.json"

    def _load_from_cache(self) -> bool:
        """Try to load filter results from cache"""
        cache_file = self._get_cache_filename()

        if not os.path.exists(cache_file):
            return False

        try:
            with open(cache_file, "r") as f:
                data = json.load(f)

            # Reconstruct statistics
            self._filter_stats = FilterStatistics()
            self._filter_stats.total_cells = data.get("total_cells", None)
            self._filter_stats.valid_cells = data.get("valid_cells", None)
            self._filter_stats.rejection_reasons = data.get("rejection_reasons", {})
            self._filter_stats.valid_cell_ids = data.get("valid_cell_ids", [])
            self._filter_stats.invalid_cells = data.get("invalid_cells_by_reason", {})

            self._valid_cell_ids = data["valid_cell_ids"]

            print(f"✓ Loaded filter results from cache: {cache_file}")
            self._filter_stats.print_summary()

            return True

        except Exception as e:
            print(f"⚠ Error loading cache: {e}")
            return False

    def _save_to_cache(self):
        """Save filter results to cache"""
        if not self.cache_results:
            return

        cache_file = self._get_cache_filename()

        try:
            data = self._filter_stats.to_dict()
            data["filters"] = self.composite_filter.get_filter_names()
            data["data_source"] = str(self.data_source.path)

            with open(cache_file, "w") as f:
                json.dump(data, f, indent=2)

            print(f"✓ Saved filter results to cache: {cache_file}")

        except Exception as e:
            print(f"⚠ Error saving cache: {e}")

    def _apply_filters(self):
        """Apply filters to all cells and cache results"""
        if self._valid_cell_ids is not None:
            return  # Already filtered

        print(f"\nApplying quality filters:")
        for filter_name in self.composite_filter.get_filter_names():
            print(f"  - {filter_name}")

        all_cell_ids = self.data_source.get_cell_ids()
        print(f"Total cells in dataset: {len(all_cell_ids):,}")

        self._filter_stats = FilterStatistics()

        for cell_id in tqdm(all_cell_ids, desc="Filtering cells"):
            try:
                cell_data = self.data_source.load_cell(cell_id)
                result = self.composite_filter(cell_data)
                self._filter_stats.record_result(cell_id, result)
            except Exception as e:
                # Record as invalid due to load error
                self._filter_stats.record_result(
                    cell_id,
                    FilterResult(
                        is_valid=False, reason="load_error", metadata={"error": str(e)}
                    ),
                )

        self._valid_cell_ids = self._filter_stats.valid_cell_ids
        self._filter_stats.print_summary()

        # Save to cache
        self._save_to_cache()

    def get_cell_ids(self) -> List[str]:
        """Get list of valid cell IDs (after filtering)"""
        if self._valid_cell_ids is None:
            self._apply_filters()
        return self._valid_cell_ids

    def load_cell(self, cell_id: str):
        """Load a cell (only valid cells allowed)"""
        if self._valid_cell_ids is None:
            self._apply_filters()

        if cell_id not in self._valid_cell_ids:
            raise ValueError(f"Cell {cell_id} is not in the valid set")

        return self.data_source.load_cell(cell_id)

    def get_statistics(self) -> FilterStatistics:
        """Get filtering statistics"""
        if self._filter_stats is None:
            self._apply_filters()
        return self._filter_stats

    def get_metadata(self):
        """Get metadata from underlying data source"""
        return self.data_source.get_metadata()

    def __len__(self) -> int:
        return len(self.get_cell_ids())
