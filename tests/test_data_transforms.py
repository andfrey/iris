"""
Tests for data_transforms module.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.data_transforms import (
    Transform,
    ChannelTransform,
    SelectPlanesTransform,
    NormalizeTransform,
    GaussianFilterTransform,
    StackChannelsTransform,
    TransformPipeline,
)
from src.data_pipeline.data_sources import CellData


class TestSelectPlanesTransform:
    """Test SelectPlanesTransform."""

    def test_select_middle_plane(self):
        """Test selecting middle plane."""
        transform = SelectPlanesTransform(plane_selection="middle")

        channels = {
            "405": [
                np.ones((250, 250)) * 1,
                np.ones((250, 250)) * 2,
                np.ones((250, 250)) * 3,
            ]
        }
        cell_data = CellData(
            cell_id="test", channels=channels, segmentation=[np.zeros((250, 250))] * 3
        )

        result = transform(cell_data)

        # Should have only 1 plane (the middle one)
        assert len(result.channels["405"]) == 1
        assert result.channels["405"][0][0, 0] == 2  # Middle plane value

    def test_select_first_plane(self):
        """Test selecting first plane."""
        transform = SelectPlanesTransform(plane_selection="first")

        channels = {
            "405": [
                np.ones((250, 250)) * 1,
                np.ones((250, 250)) * 2,
                np.ones((250, 250)) * 3,
            ]
        }
        cell_data = CellData(
            cell_id="test", channels=channels, segmentation=[np.zeros((250, 250))] * 3
        )

        result = transform(cell_data)

        assert len(result.channels["405"]) == 1
        assert result.channels["405"][0][0, 0] == 1  # First plane value

    def test_select_last_plane(self):
        """Test selecting last plane."""
        transform = SelectPlanesTransform(plane_selection="last")

        channels = {
            "405": [
                np.ones((250, 250)) * 1,
                np.ones((250, 250)) * 2,
                np.ones((250, 250)) * 3,
            ]
        }
        cell_data = CellData(
            cell_id="test", channels=channels, segmentation=[np.zeros((250, 250))] * 3
        )

        result = transform(cell_data)

        assert len(result.channels["405"]) == 1
        assert result.channels["405"][0][0, 0] == 3  # Last plane value

    def test_select_all_planes(self):
        """Test keeping all planes."""
        transform = SelectPlanesTransform(plane_selection="all")

        channels = {
            "405": [
                np.ones((250, 250)) * 1,
                np.ones((250, 250)) * 2,
                np.ones((250, 250)) * 3,
            ]
        }
        cell_data = CellData(
            cell_id="test", channels=channels, segmentation=[np.zeros((250, 250))] * 3
        )

        result = transform(cell_data)

        # Should keep all planes
        assert len(result.channels["405"]) == 3


class TestNormalizeTransform:
    """Test NormalizeTransform."""

    def test_minmax_normalization(self):
        """Test min-max normalization."""
        transform = NormalizeTransform(method="minmax", channel_keys=["405"])

        # Create channel with known range [10, 110]
        channel_data = np.ones((250, 250)) * 10
        channel_data[100:150, 100:150] = 110

        channels = {"405": [channel_data]}
        cell_data = CellData(
            cell_id="test", channels=channels, segmentation=[np.zeros((250, 250))]
        )

        result = transform(cell_data)

        # After normalization: [0, 1]
        assert result.channels["405"][0].min() >= 0
        assert result.channels["405"][0].max() <= 1
        assert result.channels["405"][0].max() > 0.9  # Max should be ~1

    def test_standardize_normalization(self):
        """Test standardization (z-score) normalization."""
        transform = NormalizeTransform(method="standardize", channel_keys=["405"])

        channel_data = np.random.randn(250, 250) * 10 + 50

        channels = {"405": [channel_data]}
        cell_data = CellData(
            cell_id="test", channels=channels, segmentation=[np.zeros((250, 250))]
        )

        result = transform(cell_data)

        # After standardization: mean~0, std~1
        normalized = result.channels["405"][0]
        assert abs(normalized.mean()) < 0.1  # Mean close to 0
        assert abs(normalized.std() - 1.0) < 0.1  # Std close to 1

    def test_multiple_channels(self):
        """Test normalizing multiple channels."""
        transform = NormalizeTransform(method="minmax", channel_keys=["405", "488"])

        channels = {
            "405": [np.ones((250, 250)) * 50],
            "488": [np.ones((250, 250)) * 100],
            "bf": [np.ones((250, 250)) * 200],  # Not normalized
        }
        cell_data = CellData(
            cell_id="test", channels=channels, segmentation=[np.zeros((250, 250))]
        )

        result = transform(cell_data)

        # 405 and 488 should be normalized
        assert "405" in result.channels
        assert "488" in result.channels
        assert "bf" in result.channels  # Still present

    def test_multiple_planes(self):
        """Test normalization with multiple planes."""
        transform = NormalizeTransform(method="minmax", channel_keys=["405"])

        channels = {
            "405": [
                np.ones((250, 250)) * 10,
                np.ones((250, 250)) * 50,
                np.ones((250, 250)) * 100,
            ]
        }
        cell_data = CellData(
            cell_id="test", channels=channels, segmentation=[np.zeros((250, 250))] * 3
        )

        result = transform(cell_data)

        # All planes should be normalized
        assert len(result.channels["405"]) == 3
        for plane in result.channels["405"]:
            assert plane.min() >= 0
            assert plane.max() <= 1


class TestGaussianFilterTransform:
    """Test GaussianFilterTransform."""

    def test_gaussian_smoothing(self):
        """Test Gaussian smoothing reduces noise."""
        transform = GaussianFilterTransform(sigma=2.0, channel_keys=["405"])

        # Create noisy image
        np.random.seed(42)
        channel_data = np.ones((250, 250)) * 100
        noise = np.random.randn(250, 250) * 20
        channel_data += noise

        channels = {"405": [channel_data.copy()]}
        cell_data = CellData(
            cell_id="test", channels=channels, segmentation=[np.zeros((250, 250))]
        )

        result = transform(cell_data)

        # Smoothed image should have less variance
        original_std = channel_data.std()
        smoothed_std = result.channels["405"][0].std()
        assert smoothed_std < original_std

    def test_different_sigma_values(self):
        """Test different sigma values."""
        channel_data = np.random.randn(250, 250) * 10 + 50

        # Small sigma
        transform_small = GaussianFilterTransform(sigma=0.5, channel_keys=["405"])
        # Large sigma
        transform_large = GaussianFilterTransform(sigma=5.0, channel_keys=["405"])

        cell_data = CellData(
            cell_id="test",
            channels={"405": [channel_data.copy()]},
            segmentation=[np.zeros((250, 250))],
        )

        result_small = transform_small(cell_data)
        result_large = transform_large(cell_data)

        # Larger sigma should smooth more (lower std)
        std_small = result_small.channels["405"][0].std()
        std_large = result_large.channels["405"][0].std()
        assert std_large < std_small


class TestStackChannelsTransform:
    """Test StackChannelsTransform."""

    def test_stack_two_channels(self):
        """Test stacking two channels."""
        transform = StackChannelsTransform(channel_order=["bf", "405"])

        channels = {"bf": [np.ones((250, 250)) * 1], "405": [np.ones((250, 250)) * 2]}
        cell_data = CellData(
            cell_id="test", channels=channels, segmentation=[np.zeros((250, 250))]
        )

        result = transform(cell_data)

        # Should have 'stacked' channel
        assert "stacked" in result.channels
        stacked = result.channels["stacked"]

        # Should be (2, 250, 250) - 2 channels stacked
        assert stacked.shape == (2, 250, 250)
        assert stacked[0, 0, 0] == 1  # bf
        assert stacked[1, 0, 0] == 2  # 405

    def test_stack_with_multiple_planes(self):
        """Test stacking channels with multiple planes."""
        transform = StackChannelsTransform(channel_order=["bf", "405"])

        channels = {
            "bf": [
                np.ones((250, 250)) * 1,
                np.ones((250, 250)) * 2,
                np.ones((250, 250)) * 3,
            ],
            "405": [
                np.ones((250, 250)) * 4,
                np.ones((250, 250)) * 5,
                np.ones((250, 250)) * 6,
            ],
        }
        cell_data = CellData(
            cell_id="test", channels=channels, segmentation=[np.zeros((250, 250))] * 3
        )

        result = transform(cell_data)

        # Should stack all planes: 2 channels × 3 planes = 6 total
        assert "stacked" in result.channels
        stacked = result.channels["stacked"]
        assert stacked.shape == (6, 250, 250)

        # Check ordering: bf_p0, bf_p1, bf_p2, 405_p0, 405_p1, 405_p2
        assert stacked[0, 0, 0] == 1  # bf plane 0
        assert stacked[1, 0, 0] == 2  # bf plane 1
        assert stacked[2, 0, 0] == 3  # bf plane 2
        assert stacked[3, 0, 0] == 4  # 405 plane 0

    def test_missing_channel_in_order(self):
        """Test error when channel in order doesn't exist."""
        transform = StackChannelsTransform(channel_order=["bf", "nonexistent"])

        channels = {"bf": [np.ones((250, 250))]}
        cell_data = CellData(
            cell_id="test", channels=channels, segmentation=[np.zeros((250, 250))]
        )

        with pytest.raises((KeyError, ValueError)):
            transform(cell_data)


class TestTransformPipeline:
    """Test TransformPipeline."""

    def test_pipeline_execution_order(self):
        """Test transforms are applied in order."""
        transforms = [
            SelectPlanesTransform(plane_selection="middle"),
            NormalizeTransform(method="minmax", channel_keys=["405"]),
            GaussianFilterTransform(sigma=1.0, channel_keys=["405"]),
        ]
        pipeline = TransformPipeline(transforms)

        channels = {
            "405": [
                np.ones((250, 250)) * 10,
                np.ones((250, 250)) * 50,
                np.ones((250, 250)) * 100,
            ]
        }
        cell_data = CellData(
            cell_id="test", channels=channels, segmentation=[np.zeros((250, 250))] * 3
        )

        result = pipeline(cell_data)

        # After SelectPlanes: 1 plane
        # After Normalize: [0, 1]
        # After Gaussian: smoothed
        assert len(result.channels["405"]) == 1
        assert (
            0 <= result.channels["405"][0].min() <= result.channels["405"][0].max() <= 1
        )

    def test_full_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline."""
        transforms = [
            SelectPlanesTransform(plane_selection="middle"),
            NormalizeTransform(method="minmax", channel_keys=["bf", "405"]),
            GaussianFilterTransform(sigma=1.0, channel_keys=["bf", "405"]),
            StackChannelsTransform(channel_order=["bf", "405"]),
        ]
        pipeline = TransformPipeline(transforms)

        channels = {
            "bf": [
                np.random.rand(250, 250) * 100,
                np.random.rand(250, 250) * 100,
                np.random.rand(250, 250) * 100,
            ],
            "405": [
                np.random.rand(250, 250) * 200,
                np.random.rand(250, 250) * 200,
                np.random.rand(250, 250) * 200,
            ],
        }
        cell_data = CellData(
            cell_id="test", channels=channels, segmentation=[np.zeros((250, 250))] * 3
        )

        result = pipeline(cell_data)

        # Final result should have stacked channels
        assert "stacked" in result.channels
        stacked = result.channels["stacked"]

        # 2 channels (after selecting middle plane)
        assert stacked.shape == (2, 250, 250)

        # Should be normalized [0, 1]
        assert stacked.min() >= 0
        assert stacked.max() <= 1

    def test_get_config(self):
        """Test getting pipeline configuration."""
        transforms = [
            SelectPlanesTransform(plane_selection="middle"),
            NormalizeTransform(method="minmax", channel_keys=["405"]),
        ]
        pipeline = TransformPipeline(transforms)

        config = pipeline.get_config()

        assert isinstance(config, dict)
        assert "transforms" in config
        assert len(config["transforms"]) == 2

    def test_empty_pipeline(self):
        """Test pipeline with no transforms."""
        pipeline = TransformPipeline([])

        channels = {"405": [np.ones((250, 250)) * 50]}
        cell_data = CellData(
            cell_id="test", channels=channels, segmentation=[np.zeros((250, 250))]
        )

        result = pipeline(cell_data)

        # Should return unchanged data
        assert result.channels["405"][0][0, 0] == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
