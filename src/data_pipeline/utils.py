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
