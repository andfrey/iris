"""
Data filtering and quality control for cell datasets.
Modular filtering system that can be composed with different criteria.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
from skimage.measure import label

CELL_DUPLICATES = {
    "10341",
    "10440",
    "10482",
    "10488",
    "10510",
    "10568",
    "10628",
    "10642",
    "10650",
    "10658",
    "10678",
    "10770",
    "10787",
    "10816",
    "10847",
    "10893",
    "10990",
    "11006",
    "11012",
    "11024",
    "11041",
    "11048",
    "11052",
    "11073",
    "11094",
    "12118",
    "12227",
    "12275",
    "12284",
    "12286",
    "12305",
    "12317",
    "12331",
    "12335",
    "12337",
    "12358",
    "12370",
    "12408",
    "12414",
    "12465",
    "12467",
    "12473",
    "12475",
    "12477",
    "12491",
    "12499",
    "12537",
    "12574",
    "12591",
    "12603",
    "12619",
    "5280",
    "5357",
    "5782",
    "5842",
    "6025",
    "6605",
    "6842",
    "6902",
    "6984",
    "6986",
    "6988",
    "7000",
    "7004",
    "7006",
    "7012",
    "7019",
    "7021",
    "7022",
    "7038",
    "7040",
    "7042",
    "7044",
    "7046",
    "7048",
    "7052",
    "7054",
    "7060",
    "7062",
    "7064",
    "7066",
    "7068",
    "7070",
    "7075",
    "7080",
    "7084",
    "7086",
    "7088",
    "7096",
    "7098",
    "7106",
    "7110",
    "7120",
    "7124",
    "7126",
    "7130",
    "7132",
    "7141",
    "7145",
    "7147",
    "7149",
    "7151",
    "7153",
    "7163",
    "7170",
    "7172",
    "7174",
    "7181",
    "7195",
    "7205",
    "7213",
    "7215",
    "7221",
    "7225",
    "7229",
    "7231",
    "7235",
    "7245",
    "7251",
    "7260",
    "7262",
    "7280",
    "7282",
    "7287",
    "7289",
    "7291",
    "7293",
    "7299",
    "7301",
    "7305",
    "7316",
    "7318",
    "7320",
    "7324",
    "7338",
    "7340",
    "7346",
    "7348",
    "7352",
    "7356",
    "7366",
    "7370",
    "7372",
    "7374",
    "7961",
    "7963",
    "7969",
    "7991",
    "7997",
    "7999",
    "8001",
    "8013",
    "8020",
    "8030",
    "8055",
    "8061",
    "8069",
    "8083",
    "8088",
    "8092",
    "8102",
    "8139",
    "8143",
    "8147",
    "8155",
    "8159",
    "8161",
    "8178",
    "8180",
    "8182",
    "8190",
    "8194",
    "8200",
    "8207",
    "8209",
    "8211",
    "8217",
    "8223",
    "8225",
    "8233",
    "8236",
    "8238",
    "8246",
    "8252",
    "8277",
    "8279",
    "8295",
    "8297",
    "8301",
    "8311",
    "8317",
    "8776",
    "8780",
    "8887",
    "9139",
    "9141",
    "9151",
    "9159",
    "9185",
    "9201",
    "9215",
    "9222",
    "9260",
    "9293",
    "9295",
    "9301",
    "9307",
    "9313",
    "9316",
    "9326",
    "9328",
    "9348",
    "9350",
    "9358",
    "9368",
    "9392",
    "9410",
    "9420",
    "9430",
    "9438",
    "9468",
    "9488",
    "9506",
    "9512",
    "9534",
    "9536",
    "9546",
    "9552",
    "9554",
    "9561",
    "9563",
    "9582",
    "9588",
    "9590",
    "9598",
    "9600",
    "9606",
    "9608",
    "9621",
    "9626",
    "9630",
    "9635",
    "9637",
    "9639",
    "9648",
    "9654",
    "9656",
    "9660",
    "9665",
    "9672",
    "9682",
    "9684",
    "9686",
    "9690",
    "9700",
    "9702",
    "9708",
    "9712",
    "9724",
    "9725",
    "9729",
    "9731",
    "9733",
    "9739",
    "9743",
    "9745",
    "9747",
    "9759",
    "9767",
    "9771",
    "9775",
    "9777",
    "9783",
    "9790",
    "9794",
    "9798",
    "9800",
    "9802",
    "9804",
    "9806",
    "9808",
    "9810",
    "9815",
    "9820",
    "9826",
    "9831",
    "9835",
    "9849",
    "9851",
    "9857",
    "9861",
    "9867",
    "9875",
    "9878",
    "9886",
    "9889",
    "9891",
    "9897",
    "9899",
}


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

    def __init__(self, expected_planes: int = 3, channels_to_check: Optional[List[str]] = None):
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

    def __init__(self, max_objects: int = 1, object_class: List[str] = ["cell"]):
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

    def __init__(
        self,
        min_pixels: int = 10,
        segmentations: List[str] = ["segmentation", "nuclei_segmentation"],
        planes_missing: str = "middle",
    ):
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
                if (
                    self.planes_missing == "last"
                    and i != len(getattr(cell_data, segmentation_attr)) - 1
                ):
                    continue
                if self.planes_missing == "middle" and (
                    i != len(getattr(cell_data, segmentation_attr)) // 2
                ):
                    continue

                num_pixels = np.sum(seg > 0)

                if num_pixels < self.min_pixels:
                    return FilterResult(
                        is_valid=False,
                        reason=f"failed_segmentation_{segmentation_attr}",
                        metadata={"num_pixels": int(num_pixels)},
                    )

        return FilterResult(is_valid=True)

    def get_name(self) -> str:
        return f"EmptySegmentation(min_pixels={self.min_pixels}, segmentations={self.segmentations}, planes_missing={self.planes_missing})"


class CellNucleiOverlappingFilter(CellFilter):
    """Filter cells where mask of the nucleus is larger than a threshold ratio of the cell mask
    indicating a likely segmentation error.
    """

    def __init__(self, max_ratio: float = 0.2, max_ratio_nuclei: float = 0.5):
        self.max_ratio = max_ratio
        self.max_ratio_nuclei = max_ratio_nuclei

    def __call__(self, cell_data) -> FilterResult:
        if cell_data.segmentation is None:
            return FilterResult(is_valid=False, reason="missing_segmentation")

        if cell_data.nuclei_segmentation is None:
            # If no nuclei segmentation, can't apply this filter
            return FilterResult(is_valid=True)

        # Get middle plane if it's a list
        seg = (
            cell_data.segmentation[len(cell_data.segmentation) // 2]
            if isinstance(cell_data.segmentation, list)
            else cell_data.segmentation
        )
        nuclei_seg = (
            cell_data.nuclei_segmentation[len(cell_data.nuclei_segmentation) // 2]
            if isinstance(cell_data.nuclei_segmentation, list)
            else cell_data.nuclei_segmentation
        )

        cell_area = np.sum(seg > 0)
        nuclei_area = np.sum(nuclei_seg > 0)
        nuclei_outside_cell = ((nuclei_seg > 0) & (seg == 0)).astype(int)
        nuclei_area_outside_cell = np.sum(nuclei_outside_cell)

        if cell_area == 0:
            return FilterResult(is_valid=False, reason="zero_cell_area")

        cell_nuclei_outside_ratio = nuclei_area_outside_cell / cell_area
        nuclei_nuclei_ouside_ratio = (
            nuclei_area_outside_cell / nuclei_area if nuclei_area > 0 else 1
        )

        if cell_nuclei_outside_ratio > self.max_ratio:
            return FilterResult(
                is_valid=False,
                reason="nuclei_too_large",
                metadata={
                    "nuclei_area_outside_cell": int(nuclei_area_outside_cell),
                    "cell_area": int(cell_area),
                    "ratio": float(cell_nuclei_outside_ratio),
                    "max_ratio": self.max_ratio,
                },
            )

        if nuclei_nuclei_ouside_ratio > self.max_ratio_nuclei:
            return FilterResult(
                is_valid=False,
                reason="nuclei_outside_cell_too_large",
                metadata={
                    "nuclei_area_outside_cell": int(nuclei_area_outside_cell),
                    "nuclei_area": int(nuclei_area),
                    "ratio": float(nuclei_nuclei_ouside_ratio),
                    "max_ratio": self.max_ratio,
                },
            )

        return FilterResult(is_valid=True)

    def get_name(self) -> str:
        return f"NucleCellNucleiOverlappingFilteriSize(max_ratio={self.max_ratio}, max_ratio_nuclei={self.max_ratio_nuclei})"


class ExpIDFilter(CellFilter):
    """Filter cells based on experiment ID."""

    # Experiment IDs that should always be excluded
    DEFAULT_EXCLUDED_EXP_IDS = ["NG012"]

    def __init__(
        self,
        allowed_exp_ids: Optional[List[str]] = None,
        excluded_exp_ids: Optional[List[str]] = None,
    ):
        """
        Args:
            allowed_exp_ids: If provided, only cells from these experiments are allowed.
            excluded_exp_ids: If provided, cells from these experiments are excluded.
                              Note: NG012 is always excluded by default.
        """
        self.allowed_exp_ids = allowed_exp_ids
        # Always include default excluded exp_ids
        excluded = set(self.DEFAULT_EXCLUDED_EXP_IDS)
        if excluded_exp_ids:
            excluded.update(excluded_exp_ids)
        self.excluded_exp_ids = list(excluded)

    def __call__(self, cell_data) -> FilterResult:
        exp_id = cell_data.metadata.get("exp_id")

        if exp_id is None:
            return FilterResult(
                is_valid=False,
                reason="missing_exp_id",
                metadata={"cell_id": cell_data.metadata.get("cell_id")},
            )

        # Check if excluded
        if exp_id in self.excluded_exp_ids:
            return FilterResult(
                is_valid=False,
                reason="excluded_exp_id",
                metadata={"exp_id": exp_id},
            )

        # Check if allowed (only if allowed_exp_ids is specified)
        if self.allowed_exp_ids is not None and exp_id not in self.allowed_exp_ids:
            return FilterResult(
                is_valid=False,
                reason="exp_id_not_allowed",
                metadata={"exp_id": exp_id, "allowed": self.allowed_exp_ids},
            )

        return FilterResult(is_valid=True)

    def get_name(self) -> str:
        parts = []
        if self.allowed_exp_ids:
            parts.append(f"allowed={self.allowed_exp_ids}")
        if self.excluded_exp_ids:
            parts.append(f"excluded={self.excluded_exp_ids}")
        return f"ExpIDFilter({', '.join(parts)})"


class CellDuplicatesFilter(CellFilter):
    """Filter cells that are known duplicates based on similarity analysis."""

    def __init__(self, duplicate_cell_ids: Optional[set] = None):
        """
        Args:
            duplicate_cell_ids: Set of cell IDs to filter out. If None, uses the
                              global CELL_DUPLICATES set defined in this module.
        """
        self.duplicate_cell_ids = (
            duplicate_cell_ids if duplicate_cell_ids is not None else CELL_DUPLICATES
        )

    def __call__(self, cell_data) -> FilterResult:
        cell_id = cell_data.metadata.get("cell_id")

        if cell_id is None:
            # Can't check without cell_id, let it pass
            return FilterResult(is_valid=True)

        # Convert to string for comparison (CELL_DUPLICATES contains strings)
        cell_id_str = str(cell_id)

        if cell_id_str in self.duplicate_cell_ids:
            return FilterResult(
                is_valid=False,
                reason="cell_duplicate",
                metadata={"cell_id": cell_id_str},
            )

        return FilterResult(is_valid=True)

    def get_name(self) -> str:
        return f"CellDuplicatesFilter(n_duplicates={len(self.duplicate_cell_ids)})"


class CellDuplicateFilter(CellFilter):
    """Filter cells that are identified as duplicates based on similarity analysis."""

    def __init__(self, duplicate_cell_ids: Optional[set] = None):
        """
        Args:
            duplicate_cell_ids: Set of cell IDs to filter out. If None, uses the
                                CELL_DUPLICATES set defined in this module.
        """
        self.duplicate_cell_ids = duplicate_cell_ids or CELL_DUPLICATES

    def __call__(self, cell_data) -> FilterResult:
        cell_id = cell_data.metadata.get("cell_id")

        if cell_id is None:
            return FilterResult(
                is_valid=False,
                reason="missing_cell_id",
            )

        # Convert to string for comparison (CELL_DUPLICATES contains strings)
        cell_id_str = str(cell_id)

        if cell_id_str in self.duplicate_cell_ids:
            return FilterResult(
                is_valid=False,
                reason="cell_duplicate",
                metadata={"cell_id": cell_id_str},
            )

        return FilterResult(is_valid=True)

    def get_name(self) -> str:
        return f"CellDuplicateFilter(n_duplicates={len(self.duplicate_cell_ids)})"


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
            for reason, count in sorted(self.rejection_reasons.items(), key=lambda x: -x[1]):
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
