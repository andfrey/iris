"""Tests for FeatureExtractor class."""

import sys
from pathlib import Path
import pytest
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.feature_extractor import FeatureExtractor


@dataclass
class MockCellData:
    """Mock CellData object for testing."""

    cell_id: str
    channels: Dict[str, List[np.ndarray]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    segmentation: Optional[List[np.ndarray]] = None
    nuclei_segmentation: Optional[List[np.ndarray]] = None


@pytest.fixture
def feature_extractor():
    """Create a FeatureExtractor instance."""
    return FeatureExtractor()


@pytest.fixture
def sample_image():
    """Create a sample intensity image."""
    # 50x50 image with varying intensities
    image = np.random.randint(0, 255, size=(50, 50), dtype=np.uint8)
    return image


@pytest.fixture
def sample_cell_mask():
    """Create a sample cell segmentation mask."""
    # Create a circular cell mask
    mask = np.zeros((50, 50), dtype=np.uint8)
    y, x = np.ogrid[:50, :50]
    center = (25, 25)
    radius = 20
    circle_mask = (x - center[1]) ** 2 + (y - center[0]) ** 2 <= radius**2
    mask[circle_mask] = 1
    return mask


@pytest.fixture
def sample_nucleus_mask():
    """Create a sample nucleus segmentation mask."""
    # Create a smaller circular nucleus mask
    mask = np.zeros((50, 50), dtype=np.uint8)
    y, x = np.ogrid[:50, :50]
    center = (25, 25)
    radius = 10
    circle_mask = (x - center[1]) ** 2 + (y - center[0]) ** 2 <= radius**2
    mask[circle_mask] = 1
    return mask


class TestExtractMorphologicalFeatures:
    """Tests for extract_morphological_features method."""

    def test_basic_morphological_features(self, feature_extractor, sample_image, sample_cell_mask):
        """Test that morphological features are extracted correctly."""
        features = feature_extractor.extract_morphological_features(
            sample_image, sample_cell_mask.copy(), "test_cell", type="cell"
        )

        # Check all expected keys are present
        expected_keys = [
            "cell_area",
            "cell_perimeter",
            "cell_mean_intensity",
            "cell_eccentricity",
            "cell_solidity",
            "cell_extent",
            "cell_major_axis_length",
            "cell_minor_axis_length",
        ]
        for key in expected_keys:
            assert key in features, f"Missing key: {key}"
            assert isinstance(features[key], (int, float)), f"Invalid type for {key}"
            assert not np.isnan(features[key]), f"NaN value for {key}"

    def test_prefix_parameter(self, feature_extractor, sample_image, sample_nucleus_mask):
        """Test that the type prefix is correctly applied."""
        features = feature_extractor.extract_morphological_features(
            sample_image, sample_nucleus_mask.copy(), "test_cell", type="nucleus"
        )

        # All keys should start with "nucleus_"
        for key in features.keys():
            assert key.startswith("nucleus_"), f"Key {key} doesn't have correct prefix"

    def test_empty_mask_raises_error(self, feature_extractor, sample_image):
        """Test that an empty mask raises ValueError."""
        empty_mask = np.zeros((50, 50), dtype=np.uint8)

        with pytest.raises(ValueError, match="No regions found in mask"):
            feature_extractor.extract_morphological_features(
                sample_image, empty_mask, "test_cell", type="cell"
            )

    def test_mask_binarization(self, feature_extractor, sample_image):
        """Test that masks with values > 1 are correctly binarized."""
        # Create mask with values > 1
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[20:30, 20:30] = 5  # Set region to 5

        features = feature_extractor.extract_morphological_features(
            sample_image, mask.copy(), "test_cell", type="cell"
        )

        # Should not raise error and should have valid area
        assert features["cell_area"] > 0

    def test_feature_values_are_positive(self, feature_extractor, sample_image, sample_cell_mask):
        """Test that extracted feature values are positive."""
        features = feature_extractor.extract_morphological_features(
            sample_image, sample_cell_mask.copy(), "test_cell", type="cell"
        )

        # Most morphological features should be positive
        for key in [
            "cell_area",
            "cell_perimeter",
            "cell_major_axis_length",
            "cell_minor_axis_length",
        ]:
            assert features[key] > 0, f"{key} should be positive"

    def test_eccentricity_bounds(self, feature_extractor, sample_image, sample_cell_mask):
        """Test that eccentricity is within [0, 1] bounds."""
        features = feature_extractor.extract_morphological_features(
            sample_image, sample_cell_mask.copy(), "test_cell", type="cell"
        )

        assert 0 <= features["cell_eccentricity"] <= 1, "Eccentricity should be between 0 and 1"

    def test_solidity_bounds(self, feature_extractor, sample_image, sample_cell_mask):
        """Test that solidity is within (0, 1] bounds."""
        features = feature_extractor.extract_morphological_features(
            sample_image, sample_cell_mask.copy(), "test_cell", type="cell"
        )

        assert 0 < features["cell_solidity"] <= 1, "Solidity should be between 0 and 1"


class TestExtractIntensityFeatures:
    """Tests for extract_intensity_features method."""

    def test_basic_intensity_features(
        self, feature_extractor, sample_image, sample_nucleus_mask, sample_cell_mask
    ):
        """Test that intensity features are extracted correctly."""
        features = feature_extractor.extract_intensity_features(
            sample_image, sample_nucleus_mask, sample_cell_mask
        )

        # Check expected keys
        assert "total_intensity_nucleus" in features
        assert "total_intensity_outside_nucleus" in features

        # Check values are non-negative
        assert features["total_intensity_nucleus"] >= 0
        assert features["total_intensity_outside_nucleus"] >= 0

    def test_cytoplasm_calculation(self, feature_extractor, sample_nucleus_mask, sample_cell_mask):
        """Test that cytoplasm intensity is correctly calculated."""
        # Create an image with uniform intensity
        image = np.ones((50, 50), dtype=np.float32) * 100

        features = feature_extractor.extract_intensity_features(
            image, sample_nucleus_mask, sample_cell_mask
        )

        # Calculate expected values
        nucleus_pixels = np.sum(sample_nucleus_mask > 0)
        cytoplasm_pixels = np.sum((sample_cell_mask > 0) & (sample_nucleus_mask == 0))

        expected_nucleus_intensity = nucleus_pixels * 100
        expected_cytoplasm_intensity = cytoplasm_pixels * 100

        assert np.isclose(features["total_intensity_nucleus"], expected_nucleus_intensity)
        assert np.isclose(features["total_intensity_outside_nucleus"], expected_cytoplasm_intensity)

    def test_no_overlap_case(self, feature_extractor):
        """Test when nucleus and cell masks don't overlap (edge case)."""
        image = np.ones((50, 50), dtype=np.float32) * 100

        # Nucleus on left side
        nucleus_mask = np.zeros((50, 50), dtype=np.uint8)
        nucleus_mask[20:30, 10:20] = 1

        # Cell on right side (no overlap)
        cell_mask = np.zeros((50, 50), dtype=np.uint8)
        cell_mask[20:30, 30:40] = 1

        features = feature_extractor.extract_intensity_features(image, nucleus_mask, cell_mask)

        # Both should have intensity
        assert features["total_intensity_nucleus"] > 0
        assert features["total_intensity_outside_nucleus"] > 0


class TestExtractAllFeatures:
    """Tests for extract_all_features method."""

    def test_extract_all_features_success(
        self, feature_extractor, sample_image, sample_nucleus_mask, sample_cell_mask
    ):
        """Test successful extraction of all features."""
        # Create mock CellData
        cell_data = MockCellData(
            cell_id="test_cell_001",
            channels={
                "405": [sample_image, sample_image, sample_image],
                "488": [sample_image, sample_image, sample_image],
                "561": [sample_image, sample_image, sample_image],
            },
            metadata={"cell_id": "test_cell_001"},
            segmentation=[sample_cell_mask, sample_cell_mask, sample_cell_mask],
            nuclei_segmentation=[sample_nucleus_mask, sample_nucleus_mask, sample_nucleus_mask],
        )

        features = feature_extractor.extract_all_features(cell_data)

        # Check that we have features from all categories
        assert len(features) > 0

        # Check for cell morphological features
        assert "cell_area" in features
        assert "cell_perimeter" in features

        # Check for nucleus morphological features
        assert "nucleus_area" in features
        assert "nucleus_perimeter" in features

        # Check for intensity features
        assert "total_intensity_nucleus" in features
        assert "total_intensity_outside_nucleus" in features

        # Check for computed features
        assert "cell_nucleus_area_ratio" in features

    def test_cell_nucleus_area_ratio(
        self, feature_extractor, sample_image, sample_nucleus_mask, sample_cell_mask
    ):
        """Test that cell_nucleus_area_ratio is correctly calculated."""
        cell_data = MockCellData(
            cell_id="test_cell_001",
            channels={
                "405": [sample_image, sample_image, sample_image],
                "488": [sample_image, sample_image, sample_image],
                "561": [sample_image, sample_image, sample_image],
            },
            metadata={"cell_id": "test_cell_001"},
            segmentation=[sample_cell_mask, sample_cell_mask, sample_cell_mask],
            nuclei_segmentation=[sample_nucleus_mask, sample_nucleus_mask, sample_nucleus_mask],
        )

        features = feature_extractor.extract_all_features(cell_data)

        # Ratio should be between 0 and 1 (nucleus smaller than cell)
        assert 0 <= features["cell_nucleus_area_ratio"] <= 1

        # Verify calculation
        expected_ratio = features["nucleus_area"] / features["cell_area"]
        assert np.isclose(features["cell_nucleus_area_ratio"], expected_ratio)

    def test_zero_cell_area_edge_case(self, feature_extractor, sample_image):
        """Test handling of zero cell area (edge case)."""
        # This should not happen in practice but test the defensive code
        empty_mask = np.zeros((50, 50), dtype=np.uint8)
        nucleus_mask = np.zeros((50, 50), dtype=np.uint8)
        nucleus_mask[20:25, 20:25] = 1

        cell_data = MockCellData(
            cell_id="test_cell_001",
            channels={
                "405": [sample_image, sample_image, sample_image],
                "488": [sample_image, sample_image, sample_image],
                "561": [sample_image, sample_image, sample_image],
            },
            metadata={"cell_id": "test_cell_001"},
            segmentation=[empty_mask, empty_mask, empty_mask],
            nuclei_segmentation=[nucleus_mask, nucleus_mask, nucleus_mask],
        )

        # Should handle gracefully (returns empty dict or dict without ratio due to exception)
        features = feature_extractor.extract_all_features(cell_data)
        # Exception should be caught and printed, so features will be empty or incomplete
        assert isinstance(features, dict)

    def test_uses_middle_plane(self, feature_extractor):
        """Test that the middle z-plane is used for feature extraction."""
        # Create 5 planes with different values
        planes = []
        for i in range(5):
            plane = np.ones((50, 50), dtype=np.uint8) * (i + 1) * 50
            planes.append(plane)

        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[20:30, 20:30] = 1

        cell_data = MockCellData(
            cell_id="test_cell_001",
            channels={
                "405": planes,
                "488": planes,
                "561": planes,
            },
            metadata={"cell_id": "test_cell_001"},
            segmentation=[mask] * 5,
            nuclei_segmentation=[mask] * 5,
        )

        features = feature_extractor.extract_all_features(cell_data)

        # The middle plane (index 2) has intensity 150
        # Mean intensity in the mask should reflect this
        assert "cell_mean_intensity" in features
        # Should be close to 150 (middle plane value)
        assert 140 <= features["cell_mean_intensity"] <= 160


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
