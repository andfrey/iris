from pathlib import Path
from typing import Dict, Any, Optional
import importlib
import os
import json
import hashlib
import pandas as pd

from sklearn.pipeline import Pipeline

from .data_sources import H5DataSource, FilteredDataSource, DataSource
from .data_filters import (
    PlaneCountFilter,
    EmptySegmentationFilter,
    MultipleObjectsFilter,
    CellNucleiOverlappingFilter,
)


def resolve(name: str):
    module_name, attr_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def create_data_source_from_config(config: Dict) -> DataSource:
    """Create DataSource from data configuration dictionary."""

    h5_path = config.get("h5_path", None)
    # Resolve relative path
    if not Path(h5_path).exists():
        abs_path = Path(__file__).resolve().parent.parent / h5_path
        if Path(abs_path).exists():
            h5_path = abs_path
        else:
            raise FileNotFoundError(f"H5 file not found: {h5_path}")

    use_quality_filters = config.get("use_quality_filters", True)
    plane_count = config.get("plane_count", 3)
    max_objects = config.get("max_objects", 1)
    min_seg_pixels = config.get("min_seg_pixels", 10)
    max_nuclei_outside_ratio = config.get("max_nuclei_outside_ratio", 1.2)
    force_refilter = config.get("force_refilter", False)
    # 1. Create data source
    print(f"\nCreating data source from: {h5_path}")
    data_source = H5DataSource(
        path=h5_path,
        plane_selection="all",  # Load all planes, select later in transform
    )
    print(f"   ✓ Found {len(data_source.get_cell_ids()):,} total cells")

    # 2. Apply quality filters (optional)
    if use_quality_filters:
        print(f"\nApplying quality filters:")
        print(f"   - Plane count: {plane_count}")
        print(f"   - Max objects: {max_objects}")
        print(f"   - Min segmentation pixels: {min_seg_pixels}")
        print(f"   - Max nuclei outside ratio: {max_nuclei_outside_ratio}")

        filters = [
            PlaneCountFilter(expected_planes=plane_count),
            EmptySegmentationFilter(min_pixels=min_seg_pixels),
            MultipleObjectsFilter(max_objects=max_objects),
            CellNucleiOverlappingFilter(max_ratio=max_nuclei_outside_ratio),
        ]

        filtered_source = FilteredDataSource(
            data_source=data_source,
            filters=filters,
            cache_results=True,
            force_refilter=force_refilter,
        )

        print(f"   ✓ Filtered to {len(filtered_source.get_cell_ids()):,} valid cells")

        data_source = filtered_source
    else:
        print(f"Skipping quality filters")

    return data_source


def create_transform_pipeline_from_config(config: Dict, transform_type: str = "image"):
    # Local import to avoid circular import
    from .data_transforms import TransformPipeline

    if transform_type not in ["image", "feature"]:
        raise ValueError(f"Invalid transform pipeline type: {transform_type}")
    print(f"\nCreating {transform_type} transform pipeline")
    print(f"   - Using {config.get(f'{transform_type}_transform_config', [])} transforms")

    transforms = [
        resolve(cfg["class_path"])(**cfg.get("init_args", {}))
        for cfg in config.get(f"{transform_type}_transform_config", [])
    ]
    if transform_type == "image":
        transform_pipeline = TransformPipeline(transforms)
    else:
        from sklearn.pipeline import Pipeline

        transform_pipeline = Pipeline(
            steps=[(repr(transform), transform) for transform in transforms]
        )

    return transform_pipeline


# ---------------------
# Cache helper utilities
# ---------------------


def save_parquet_cache(df: pd.DataFrame, cache_file: str):
    try:
        df.to_parquet(cache_file, index=False)
        print(f"✓ Saved cache: {cache_file}")
    except Exception as e:
        print(f"⚠ Error saving cache: {e}")


def load_parquet_cache(cache_file: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(cache_file):
        return None
    try:
        df = pd.read_parquet(cache_file)
        print(f"✓ Loaded cache: {cache_file}")
        return df
    except Exception as e:
        print(f"⚠ Error loading cache: {e}")
        return None


def make_hash_from_dict(d: dict, length: int = 12) -> str:
    cache_str = json.dumps(d, sort_keys=True)
    return hashlib.md5(cache_str.encode()).hexdigest()[:length]
