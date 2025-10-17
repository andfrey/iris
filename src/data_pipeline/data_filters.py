"""
Data filtering and quality control for cell datasets.
Modular filtering system that can be composed with different criteria.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
from skimage.measure import label


@dataclass
class FilterResult:
    """Result of applying a filter to a cell"""

    is_valid: bool
    reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class CellFilter(ABC):
    """Base class for cell quality filters"""

    @abstractmethod
    def __call__(self, cell_data) -> FilterResult:
        """
        Apply filter to cell data.

        Args:
            cell_data: CellData object to filter

        Returns:
            FilterResult with is_valid flag and optional reason/metadata
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return filter name for logging"""
        pass


class PlaneCountFilter(CellFilter):
    """Filter cells that don't have the expected number of planes per channel"""

    def __init__(
        self, expected_planes: int = 3, channels_to_check: Optional[List[str]] = None
    ):
        self.expected_planes = expected_planes
        self.channels_to_check = channels_to_check or ["405", "488", "561", "bf"]

    def __call__(self, cell_data) -> FilterResult:
        for channel in self.channels_to_check:
            if channel not in cell_data.channels:
                # Channel not present - skip this check
                continue

            planes = cell_data.channels[channel]
            n_planes = len(planes) if isinstance(planes, list) else 1

            if n_planes != self.expected_planes:
                return FilterResult(
                    is_valid=False,
                    reason="invalid_plane_count",
                    metadata={
                        "channel": channel,
                        "expected": self.expected_planes,
                        "actual": n_planes,
                    },
                )

        return FilterResult(is_valid=True)

    def get_name(self) -> str:
        return f"PlaneCount(expected={self.expected_planes})"


class MultipleObjectsFilter(CellFilter):
    """Filter cells with multiple objects in segmentation mask"""
    object_class_mask_map = {
        "cell": "segmentation",
        "nuclei": "nuclei_segmentation",
    }
    def __init__(self, max_objects: int = 1, object_class: List[str] = ["cell", "nuclei"]):
        self.max_objects = max_objects
        self.object_masks = [self.object_class_mask_map[cls] for cls in object_class]

    def __call__(self, cell_data) -> FilterResult:
        if cell_data.segmentation is None:
            return FilterResult(is_valid=False, reason="missing_segmentation")
        for mask_attr in self.object_masks:
            if not getattr(cell_data, mask_attr):
                return FilterResult(is_valid=False, reason="missing_segmentation")
            for i, seg in enumerate(getattr(cell_data, mask_attr)):
                # Label connected components
                labeled_mask = label(seg)
                num_objects = labeled_mask.max()

                if num_objects > self.max_objects:
                    return FilterResult(
                        is_valid=False,
                        reason="multiple_cells",
                        metadata={
                            "num_objects": int(num_objects),
                            "object_class": mask_attr,
                            "max_allowed": self.max_objects,
                            "plane_index": i,
                        },
                    )

        return FilterResult(is_valid=True)

    def get_name(self) -> str:
        return f"MultipleObjects(max={self.max_objects}, object_classes={self.object_masks})"


class EmptySegmentationFilter(CellFilter):
    """Filter cells where segmentation failed (empty mask)"""

    def __init__(self, min_pixels: int = 10, segmentations: List[str] = ["segmentation", "nuclei_segmentation"], planes_missing: str = "middle"):
        self.min_pixels = min_pixels
        self.segmentations = segmentations
        self.planes_missing = planes_missing

    def __call__(self, cell_data) -> FilterResult:
        for segmentation_attr in self.segmentations:
            if not getattr(cell_data, segmentation_attr):
                return FilterResult(is_valid=False, reason="missing_segmentation")
            
            for i, seg in enumerate(getattr(cell_data, segmentation_attr)):
                if self.planes_missing == "first" and i != 0:
                    continue
                if self.planes_missing == "last" and i != len(getattr(cell_data, segmentation_attr)) - 1:
                    continue
                if self.planes_missing == "middle" and (i != len(getattr(cell_data, segmentation_attr)) // 2):
                    continue
                    
                num_pixels = np.sum(seg > 0)
                
                if num_pixels < self.min_pixels:
                    return FilterResult(
                        is_valid=False,
                        reason="failed_segmentation",
                        metadata={"num_pixels": int(num_pixels)},
                    )
            
        return FilterResult(is_valid=True)

    def get_name(self) -> str:
        return f"EmptySegmentation(min_pixels={self.min_pixels}, segmentations={self.segmentations}, planes_missing={self.planes_missing})"


class NucleiSizeFilter(CellFilter):
    """Filter cells where mask of the nucleus is larger than a threshold ratio of the cell mask
    indicating a likely segmentation error.
    """

    def __init__(self, max_ratio: float = 1.2):
        self.max_ratio = max_ratio

    def __call__(self, cell_data) -> FilterResult:
        if cell_data.segmentation is None:
            return FilterResult(is_valid=False, reason="missing_segmentation")

        if cell_data.nuclei_segmentation is None:
            # If no nuclei segmentation, can't apply this filter
            return FilterResult(is_valid=True)

        # Get first plane if it's a list
        seg = (
            cell_data.segmentation[0]
            if isinstance(cell_data.segmentation, list)
            else cell_data.segmentation
        )
        nuclei_seg = (
            cell_data.nuclei_segmentation[0]
            if isinstance(cell_data.nuclei_segmentation, list)
            else cell_data.nuclei_segmentation
        )

        cell_area = np.sum(seg > 0)
        nuclei_area = np.sum(nuclei_seg > 0)

        if cell_area == 0:
            return FilterResult(is_valid=False, reason="zero_cell_area")

        ratio = nuclei_area / cell_area

        if ratio > self.max_ratio:
            return FilterResult(
                is_valid=False,
                reason="nuclei_too_large",
                metadata={
                    "nuclei_area": int(nuclei_area),
                    "cell_area": int(cell_area),
                    "ratio": float(ratio),
                    "max_ratio": self.max_ratio,
                },
            )

        return FilterResult(is_valid=True)

    def get_name(self) -> str:
        return f"NucleiSize(max_ratio={self.max_ratio})"


class CompositeFilter:
    """Combines multiple filters"""

    def __init__(self, filters: List[CellFilter]):
        self.filters = filters

    def __call__(self, cell_data) -> FilterResult:
        """Apply all filters. Returns first failure or success if all pass."""
        for filter_obj in self.filters:
            result = filter_obj(cell_data)
            if not result.is_valid:
                return result

        return FilterResult(is_valid=True)

    def get_filter_names(self) -> List[str]:
        """Get names of all filters"""
        return [f.get_name() for f in self.filters]


class FilterStatistics:
    """Tracks statistics about filtering"""

    def __init__(self):
        self.total_cells = 0
        self.valid_cells = 0
        self.rejection_reasons: Dict[str, int] = {}
        self.valid_cell_ids: List[str] = []
        self.invalid_cells: Dict[str, List[str]] = {}

    def record_result(self, cell_id: str, result: FilterResult):
        """Record the result of filtering a cell"""
        self.total_cells += 1

        if result.is_valid:
            self.valid_cells += 1
            self.valid_cell_ids.append(cell_id)
        else:
            reason = result.reason or "unknown"
            self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1

            if reason not in self.invalid_cells:
                self.invalid_cells[reason] = []
            self.invalid_cells[reason].append(cell_id)

    def print_summary(self):
        """Print filtering statistics"""
        print("\n" + "=" * 60)
        print("FILTERING STATISTICS")
        print("=" * 60)
        print(f"Total cells:                {self.total_cells:,}")
        print(f"Valid cells:                {self.valid_cells:,}")
        print(f"Invalid cells:              {self.total_cells - self.valid_cells:,}")

        if self.rejection_reasons:
            print(f"\nRejection reasons:")
            for reason, count in sorted(
                self.rejection_reasons.items(), key=lambda x: -x[1]
            ):
                print(f"  - {reason:30s}: {count:,}")

        if self.total_cells > 0:
            retention = 100 * self.valid_cells / self.total_cells
            print(f"\nRetention rate:             {retention:.1f}%")
        print("=" * 60)

    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary"""
        return {
            "total_cells": self.total_cells,
            "valid_cells": self.valid_cells,
            "invalid_cells": self.total_cells - self.valid_cells,
            "rejection_reasons": self.rejection_reasons,
            "retention_rate": 100 * self.valid_cells / max(self.total_cells, 1),
            "valid_cell_ids": self.valid_cell_ids,
            "invalid_cells_by_reason": self.invalid_cells,
        }
