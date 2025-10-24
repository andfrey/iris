"""Tests for ModularCellFeaturesDataset class."""

import sys
import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.dataset import ModularCellFeaturesDataset, compute_fucci_labels
from src.data_pipeline.data_sources import CellData

np.random.seed(42)


@dataclass
class MockCellData:
    """Mock CellData object for testing."""

    cell_id: str
    channels: Dict[str, List[np.ndarray]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    segmentation: Optional[List[np.ndarray]] = None
    nuclei_segmentation: Optional[List[np.ndarray]] = None


class MockDataSource:
    """Mock DataSource for testing."""

    def __init__(self, num_cells=10):
        self.num_cells = num_cells
        self.path = Path("/mock/path/data.h5")

    def get_cell_ids(self) -> List[str]:
        return [f"cell_{i:03d}" for i in range(self.num_cells)]

    def load_cell(self, cell_id: str) -> MockCellData:
        # Create consistent test data
        image = np.random.randint(50, 200, size=(50, 50), dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[20:30, 20:30] = 1

        nucleus_mask = np.zeros((50, 50), dtype=np.uint8)
        nucleus_mask[23:27, 23:27] = 1

        return MockCellData(
            cell_id=cell_id,
            channels={
                "405": [image, image, image],
                "488": [image, image, image],
                "561": [image, image, image],
            },
            metadata={"cell_id": cell_id},
            segmentation=[mask, mask, mask],
            nuclei_segmentation=[nucleus_mask, nucleus_mask, nucleus_mask],
        )


@pytest.fixture
def mock_config():
    """Create a mock configuration dictionary."""
    return {
        "h5_path": "data/test.h5",
        "seed": 42,
        "train_test_ratio": 0.8,
        "use_quality_filters": False,
    }


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_dataset(mock_config):
    """Create a ModularCellFeaturesDataset with mocked dependencies."""
    with (
        patch("src.data_pipeline.dataset.create_data_source_from_config") as mock_source,
        patch("src.data_pipeline.dataset.create_transform_pipeline_from_config") as mock_transform,
    ):
        mock_source.return_value = MockDataSource(num_cells=20)
        mock_transform.return_value = None

        dataset = ModularCellFeaturesDataset(
            data_config=mock_config,
            mask_intensity="segmentation",
            use_cache=False,  # Disable cache for most tests
        )

        return dataset


class TestModularCellFeaturesDatasetInit:
    """Tests for dataset initialization."""

    def test_init_creates_data_source(self, mock_config):
        """Test that initialization creates a data source."""
        with (
            patch("src.data_pipeline.dataset.create_data_source_from_config") as mock_source,
            patch(
                "src.data_pipeline.dataset.create_transform_pipeline_from_config"
            ) as mock_transform,
        ):
            mock_source.return_value = MockDataSource(num_cells=10)
            mock_transform.return_value = None

            dataset = ModularCellFeaturesDataset(data_config=mock_config, use_cache=False)

            mock_source.assert_called_once_with(mock_config)
            assert len(dataset) == 10

    def test_init_creates_transforms(self, mock_config):
        """Test that initialization creates transform pipelines."""
        with (
            patch("src.data_pipeline.dataset.create_data_source_from_config") as mock_source,
            patch(
                "src.data_pipeline.dataset.create_transform_pipeline_from_config"
            ) as mock_transform,
        ):
            mock_source.return_value = MockDataSource(num_cells=10)
            mock_transform.return_value = None

            dataset = ModularCellFeaturesDataset(data_config=mock_config, use_cache=False)

            # Should be called twice: once for image, once for feature
            assert mock_transform.call_count == 2
            calls = mock_transform.call_args_list
            assert calls[0][1]["transform_type"] == "image"
            assert calls[1][1]["transform_type"] == "feature"

    def test_init_stores_config(self, mock_config):
        """Test that configuration is stored."""
        with (
            patch("src.data_pipeline.dataset.create_data_source_from_config") as mock_source,
            patch("src.data_pipeline.dataset.create_transform_pipeline_from_config"),
        ):
            mock_source.return_value = MockDataSource(num_cells=10)

            dataset = ModularCellFeaturesDataset(data_config=mock_config, use_cache=False)

            assert dataset.config == mock_config
            assert dataset.mask_intensity == "segmentation"
            assert dataset.use_cache == False


class TestModularCellFeaturesDatasetBasics:
    """Tests for basic dataset operations."""

    def test_len(self, mock_dataset):
        """Test __len__ returns correct number of samples."""
        assert len(mock_dataset) == 20

    def test_getitem_returns_tuple(self, mock_dataset):
        """Test __getitem__ returns (features, labels) tuple."""
        features, labels = mock_dataset[0]

        assert isinstance(features, dict)
        assert isinstance(labels, np.ndarray)
        assert labels.shape == (2,)

    def test_getitem_extracts_features(self, mock_dataset):
        """Test that features are extracted correctly."""
        features, labels = mock_dataset[0]

        # Check for morphological features
        assert "cell_area" in features
        assert "nucleus_area" in features
        assert "cell_perimeter" in features

        # Check for intensity features
        assert "total_intensity_nucleus" in features
        assert "total_intensity_outside_nucleus" in features

        # Check for computed features
        assert "cell_nucleus_area_ratio" in features

    def test_getitem_computes_labels(self, mock_dataset):
        """Test that FUCCI labels are computed."""
        features, labels = mock_dataset[0]

        # Labels should be non-negative
        assert labels[0] >= 0  # 488 intensity
        assert labels[1] >= 0  # 561 intensity

    def test_getitem_different_indices(self, mock_dataset):
        """Test that different indices return data."""
        features0, labels0 = mock_dataset[0]
        features1, labels1 = mock_dataset[1]

        # Both should return valid data
        assert isinstance(features0, dict)
        assert isinstance(features1, dict)
        assert len(features0) > 0
        assert len(features1) > 0


class TestGetDatasetDF:
    """Tests for get_dataset_df method."""

    def test_get_dataset_df_returns_dataframe(self, mock_dataset):
        """Test that get_dataset_df returns a DataFrame."""
        df = mock_dataset.get_dataset_df()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_get_dataset_df_has_labels(self, mock_dataset):
        """Test that DataFrame contains label columns."""
        df = mock_dataset.get_dataset_df()

        assert "label_488" in df.columns
        assert "label_561" in df.columns

    def test_get_dataset_df_has_features(self, mock_dataset):
        """Test that DataFrame contains feature columns."""
        df = mock_dataset.get_dataset_df()

        # Should have morphological features
        assert "cell_area" in df.columns
        assert "nucleus_area" in df.columns

        # Should have intensity features
        assert "total_intensity_nucleus" in df.columns

    def test_get_dataset_df_correct_length(self, mock_dataset):
        """Test that DataFrame has correct number of rows."""
        df = mock_dataset.get_dataset_df()

        assert len(df) == len(mock_dataset)

    def test_get_dataset_df_no_missing_values(self, mock_dataset):
        """Test that DataFrame has no missing required values."""
        df = mock_dataset.get_dataset_df()

        # Labels should never be missing
        assert not df["label_488"].isna().any()
        assert not df["label_561"].isna().any()


class TestCachingMechanism:
    """Tests for feature caching functionality."""

    def test_cache_key_generation(self, mock_dataset):
        """Test that cache keys are generated."""
        cache_key = mock_dataset._get_cache_key()

        assert isinstance(cache_key, str)
        assert cache_key.startswith("features_")
        assert len(cache_key) > len("features_")

    def test_cache_key_deterministic(self, mock_dataset):
        """Test that cache keys are deterministic."""
        key1 = mock_dataset._get_cache_key()
        key2 = mock_dataset._get_cache_key()

        assert key1 == key2

    def test_cache_dir_creation(self, mock_dataset, temp_cache_dir, monkeypatch):
        """Test that cache directory is created."""
        # Monkeypatch the data source path
        mock_dataset.data_source.path = temp_cache_dir / "data.h5"

        cache_dir = mock_dataset._get_cache_dir()

        assert cache_dir.exists()
        assert cache_dir.name == "features"

    def test_cache_path_generation(self, mock_dataset, temp_cache_dir, monkeypatch):
        """Test that cache path is generated correctly."""
        mock_dataset.data_source.path = temp_cache_dir / "data.h5"

        cache_path = mock_dataset._get_cache_path()

        assert cache_path.suffix == ".parquet"
        assert "features_" in cache_path.stem

    def test_save_and_load_cache(self, mock_dataset, temp_cache_dir, monkeypatch):
        """Test saving and loading cached DataFrame."""
        mock_dataset.data_source.path = temp_cache_dir / "data.h5"
        mock_dataset.use_cache = True

        # Create a sample DataFrame
        df = pd.DataFrame(
            {
                "cell_area": [100, 200],
                "nucleus_area": [50, 100],
                "label_488": [10.0, 20.0],
                "label_561": [15.0, 25.0],
            }
        )

        # Save to cache
        mock_dataset._save_cached_df(df)

        # Load from cache
        loaded_df = mock_dataset._load_cached_df()

        assert loaded_df is not None
        assert len(loaded_df) == len(df)
        assert list(loaded_df.columns) == list(df.columns)
        pd.testing.assert_frame_equal(loaded_df, df)

    def test_cache_disabled_returns_none(self, mock_dataset):
        """Test that caching can be disabled."""
        mock_dataset.use_cache = False

        loaded_df = mock_dataset._load_cached_df()

        assert loaded_df is None


class TestSplitMethods:
    """Tests for data splitting methods."""

    def test_split_X_y(self, mock_dataset):
        """Test split_X_y separates features and labels."""
        df = mock_dataset.get_dataset_df()
        X, y = mock_dataset.split_X_y(df)

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

        # X should have features (all columns except labels)
        assert X.shape[0] == len(df)
        assert X.shape[1] == len(df.columns) - 2  # Minus 2 label columns

        # y should have 2 columns (488 and 561)
        assert y.shape == (len(df), 2)

    def test_split_X_y_values(self, mock_dataset):
        """Test that split_X_y preserves values correctly."""
        df = mock_dataset.get_dataset_df()
        X, y = mock_dataset.split_X_y(df)

        # Check label values match
        assert np.allclose(y[:, 0], df["label_488"].values)
        assert np.allclose(y[:, 1], df["label_561"].values)

    def test_split_train_test_set(self, mock_dataset):
        """Test train/test splitting."""
        train_df, test_df = mock_dataset.split_train_test_set()

        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)

        # Check sizes
        total_size = len(train_df) + len(test_df)
        assert total_size == len(mock_dataset)

        # Check train is larger (default 0.9 split)
        assert len(train_df) > len(test_df)

    def test_split_train_test_set_ratio(self, mock_config):
        """Test that train/test ratio is respected."""
        mock_config["train_test_ratio"] = 0.7
        mock_config["image_transform_config"] = []
        mock_config["feature_transform_config"] = []

        with (patch("src.data_pipeline.dataset.create_data_source_from_config") as mock_source,):
            mock_source.return_value = MockDataSource(num_cells=100)

            dataset = ModularCellFeaturesDataset(data_config=mock_config, use_cache=False)

            train_df, test_df = dataset.split_train_test_set()

            train_ratio = len(train_df) / (len(train_df) + len(test_df))
            assert abs(train_ratio - 0.7) < 0.05  # Allow small deviation

    def test_split_reproducibility(self, mock_dataset):
        """Test that splits are reproducible with same seed."""
        train_df1, test_df1 = mock_dataset.split_train_test_set()
        train_df2, test_df2 = mock_dataset.split_train_test_set()

        # Should get same splits (same seed in config)
        assert len(train_df1) == len(train_df2)
        assert len(test_df1) == len(test_df2)


class TestTransformIntegration:
    """Tests for transform pipeline integration."""

    def test_image_transform_applied(self, mock_config):
        """Test that image transforms are applied."""
        mock_transform = Mock()
        mock_transform.side_effect = lambda x: x  # Pass through

        with (
            patch("src.data_pipeline.dataset.create_data_source_from_config") as mock_source,
            patch("src.data_pipeline.dataset.create_transform_pipeline_from_config") as mock_tf,
        ):
            mock_source.return_value = MockDataSource(num_cells=5)

            def transform_factory(config, transform_type):
                if transform_type == "image":
                    return mock_transform
                return None

            mock_tf.side_effect = transform_factory

            dataset = ModularCellFeaturesDataset(data_config=mock_config, use_cache=False)

            # Get one sample
            features, labels = dataset[0]

            # Image transform should have been called
            assert mock_transform.called

    def test_feature_transform_in_get_dataset_df(self, mock_config):
        """Test that feature transforms are applied in get_dataset_df."""
        mock_feature_transform = Mock()
        mock_feature_transform.fit = Mock()
        mock_feature_transform.transform = Mock(side_effect=lambda x: x.values)

        with (
            patch("src.data_pipeline.dataset.create_data_source_from_config") as mock_source,
            patch("src.data_pipeline.dataset.create_transform_pipeline_from_config") as mock_tf,
        ):
            mock_source.return_value = MockDataSource(num_cells=5)

            def transform_factory(config, transform_type):
                if transform_type == "feature":
                    return mock_feature_transform
                return None

            mock_tf.side_effect = transform_factory

            dataset = ModularCellFeaturesDataset(data_config=mock_config, use_cache=False)

            df = dataset.get_dataset_df()

            # Feature transform should have been fitted and applied
            assert mock_feature_transform.fit.called
            assert mock_feature_transform.transform.called

    def test_standard_scaler_applied(self, mock_config):
        """Test that StandardScaler is applied to features."""
        from sklearn.preprocessing import StandardScaler

        # Add StandardScaler to config
        mock_config["feature_transform_config"] = [
            {"class_path": "sklearn.preprocessing.StandardScaler", "init_args": {}}
        ]

        with (patch("src.data_pipeline.dataset.create_data_source_from_config") as mock_source,):
            mock_source.return_value = MockDataSource(num_cells=10)

            dataset = ModularCellFeaturesDataset(data_config=mock_config, use_cache=False)

            df = dataset.get_dataset_df()

            # Check that features are standardized (mean ~ 0, std ~ 1)
            feature_cols = [col for col in df.columns if col not in ["label_488", "label_561"]]

            # All feature columns should exist
            assert len(feature_cols) > 0

            # For each feature, mean should be close to 0 and std close to 1
            for col in feature_cols:
                mean = df[col].mean()
                std = df[col].std()
                # Allow some tolerance due to small sample size
                assert abs(mean) < 1e-4, f"{col} mean {mean} not close to 0"
                assert abs(std - 1.0) < 0.1 or std == 0.0, f"{col} std {std} not close to 1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
