"""
Tests for data_transforms module.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.data_transforms import (
    SelectPlanesTransform,
    NormalizeTransform,
    GaussianFilterTransform,
    TransformPipeline,
    FUCCIScaleTransform,
    RemoveBackgroundTransform,
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


class TestFUCCIScaleTransform:
    """Test FUCCIScaleTransform."""

    def test_default_scale_factors(self):
        """Test scaling with default scale factors."""
        transform = FUCCIScaleTransform()

        # Create test data with known values
        channels = {
            "488": [np.ones((250, 250)) * 18000],  # Should scale to 1.0
            "561": [np.ones((250, 250)) * 40000],  # Should scale to 1.0
        }
        cell_data = CellData(cell_id="test", channels=channels, segmentation=[np.zeros((250, 250))])

        result = transform(cell_data)

        # Check scaling applied correctly
        assert np.all(result.channels["488"][0] == 1.0)
        assert np.all(result.channels["561"][0] == 1.0)

    def test_custom_scale_factors(self):
        """Test scaling with custom scale factors."""
        transform = FUCCIScaleTransform(scale_divider_488=10000, scale_divider_561=20000)

        channels = {
            "488": [np.ones((250, 250)) * 10000],  # Should scale to 1.0
            "561": [np.ones((250, 250)) * 20000],  # Should scale to 1.0
        }
        cell_data = CellData(cell_id="test", channels=channels, segmentation=[np.zeros((250, 250))])

        result = transform(cell_data)

        assert np.all(result.channels["488"][0] == 1.0)
        assert np.all(result.channels["561"][0] == 1.0)

    def test_scale_488_channel_only(self):
        """Test scaling only 488 channel."""
        transform = FUCCIScaleTransform(channel_keys=["488"])

        channels = {
            "488": [np.ones((250, 250)) * 18000],
            "561": [np.ones((250, 250)) * 40000],  # Should NOT be scaled
        }
        cell_data = CellData(cell_id="test", channels=channels, segmentation=[np.zeros((250, 250))])

        result = transform(cell_data)

        # 488 should be scaled
        assert np.allclose(result.channels["488"][0], 1.0)
        # 561 should remain unchanged
        assert np.allclose(result.channels["561"][0], 40000)

    def test_scale_multiple_planes(self):
        """Test scaling with multiple planes."""
        transform = FUCCIScaleTransform()

        channels = {
            "488": [
                np.ones((250, 250)) * 18000,
                np.ones((250, 250)) * 36000,
                np.ones((250, 250)) * 9000,
            ],
            "561": [
                np.ones((250, 250)) * 40000,
                np.ones((250, 250)) * 80000,
                np.ones((250, 250)) * 20000,
            ],
        }
        cell_data = CellData(
            cell_id="test", channels=channels, segmentation=[np.zeros((250, 250))] * 3
        )

        result = transform(cell_data)

        # Check all planes are scaled
        assert len(result.channels["488"]) == 3
        assert len(result.channels["561"]) == 3

        # Check specific values
        assert np.allclose(result.channels["488"][0], 1.0)
        assert np.allclose(result.channels["488"][1], 2.0)
        assert np.allclose(result.channels["488"][2], 0.5)

        assert np.allclose(result.channels["561"][0], 1.0)
        assert np.allclose(result.channels["561"][1], 2.0)
        assert np.allclose(result.channels["561"][2], 0.5)

    def test_scale_with_real_intensities(self):
        """Test scaling with realistic intensity values."""
        transform = FUCCIScaleTransform()

        # Realistic intensity ranges
        channels = {
            "488": [np.random.randint(5000, 30000, size=(250, 250)).astype(float)],
            "561": [np.random.randint(10000, 60000, size=(250, 250)).astype(float)],
        }
        cell_data = CellData(cell_id="test", channels=channels, segmentation=[np.zeros((250, 250))])

        result = transform(cell_data)

        # Scaled values should be in reasonable range
        assert result.channels["488"][0].min() > 0
        assert result.channels["488"][0].max() < 2.0  # ~30000 / 18000
        assert result.channels["561"][0].min() > 0
        assert result.channels["561"][0].max() < 2.0  # ~60000 / 40000

    def test_scale_preserves_other_channels(self):
        """Test that non-FUCCI channels are not modified."""
        transform = FUCCIScaleTransform()

        channels = {
            "488": [np.ones((250, 250)) * 18000],
            "561": [np.ones((250, 250)) * 40000],
            "bf": [np.ones((250, 250)) * 1000],  # Should remain unchanged
            "405": [np.ones((250, 250)) * 5000],  # Should remain unchanged
        }
        cell_data = CellData(cell_id="test", channels=channels, segmentation=[np.zeros((250, 250))])

        result = transform(cell_data)

        # FUCCI channels scaled
        assert np.allclose(result.channels["488"][0], 1.0)
        assert np.allclose(result.channels["561"][0], 1.0)
        # Other channels unchanged
        assert np.allclose(result.channels["bf"][0], 1000)
        assert np.allclose(result.channels["405"][0], 5000)

    def test_invalid_channel_error(self):
        """Test error when scaling unsupported channel."""
        # Create transform with unsupported channel
        transform = FUCCIScaleTransform(channel_keys=["unsupported_channel"])

        channels = {
            "unsupported_channel": [np.ones((250, 250)) * 1000],
        }
        cell_data = CellData(cell_id="test", channels=channels, segmentation=[np.zeros((250, 250))])

        # Should raise ValueError for undefined scale factor
        with pytest.raises(ValueError, match="No scale factor defined"):
            transform(cell_data)

    def test_get_config(self):
        """Test getting transform configuration."""
        transform = FUCCIScaleTransform(scale_divider_488=15000, scale_divider_561=35000)

        config = transform.get_config()

        assert config["type"] == "FUCCIScaleTransform"
        assert "channel_keys" in config
        assert "scale_factors" in config
        assert config["scale_factors"]["488"] == 1.0 / 15000
        assert config["scale_factors"]["561"] == 1.0 / 35000

    def test_scale_in_pipeline(self):
        """Test FUCCIScaleTransform in a pipeline."""
        transforms = [
            SelectPlanesTransform(plane_selection="middle"),
            FUCCIScaleTransform(),
            NormalizeTransform(method="minmax", channel_keys=["488", "561"]),
        ]
        pipeline = TransformPipeline(transforms)

        channels = {
            "488": [
                np.ones((250, 250)) * 9000,
                np.ones((250, 250)) * 18000,
                np.ones((250, 250)) * 27000,
            ],
            "561": [
                np.ones((250, 250)) * 20000,
                np.ones((250, 250)) * 40000,
                np.ones((250, 250)) * 60000,
            ],
        }
        cell_data = CellData(
            cell_id="test", channels=channels, segmentation=[np.zeros((250, 250))] * 3
        )

        result = pipeline(cell_data)

        # After SelectPlanes: middle plane selected
        # After FUCCIScale: values scaled
        # After Normalize: values in [0, 1]
        assert len(result.channels["488"]) == 1
        assert len(result.channels["561"]) == 1
        assert 0 <= result.channels["488"][0].min() <= result.channels["488"][0].max() <= 1
        assert 0 <= result.channels["561"][0].min() <= result.channels["561"][0].max() <= 1


class TestNormalizeTransform:
    """Test NormalizeTransform."""

    def test_minmax_normalization(self):
        """Test min-max normalization."""
        transform = NormalizeTransform(method="minmax", channel_keys=["405"])

        # Create channel with known range [10, 110]
        channel_data = np.ones((250, 250)) * 10
        channel_data[100:150, 100:150] = 110

        channels = {"405": [channel_data]}
        cell_data = CellData(cell_id="test", channels=channels, segmentation=[np.zeros((250, 250))])

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
        cell_data = CellData(cell_id="test", channels=channels, segmentation=[np.zeros((250, 250))])

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
        cell_data = CellData(cell_id="test", channels=channels, segmentation=[np.zeros((250, 250))])

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
        cell_data = CellData(cell_id="test", channels=channels, segmentation=[np.zeros((250, 250))])

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


def make_image(shape=(10, 12), fill=100.0):
    return np.full(shape, fill, dtype=float)


def make_mask(shape=(10, 12), on_region=(slice(2, 8), slice(3, 9))):
    m = np.zeros(shape, dtype=np.uint8)
    m[on_region] = 1
    return m


class DummyData:
    def __init__(self, channels, segmentation=None, nuclei_segmentation=None):
        # channels: dict of key -> either single ndarray or list of ndarrays
        self.channels = channels
        self.segmentation = segmentation
        self.nuclei_segmentation = nuclei_segmentation


def test_remove_background_multiplane_uses_masks_per_plane():
    # create two planes and two masks
    img1 = make_image()
    img2 = make_image(fill=50.0)
    mask1 = make_mask()
    # second mask zeros out everything (should zero-out whole image)
    mask2 = make_mask(on_region=(slice(1, 2), slice(0, 1)))

    data = DummyData(channels={"488": [img1.copy(), img2.copy()]}, segmentation=[mask1, mask2])

    t = RemoveBackgroundTransform(channel_keys=["488"], background_padding=0, mask="cell")
    out = t(data)

    # first plane: pixels outside mask1 should be zero
    out1 = out.channels["488"][0]
    assert np.all(out1[mask1 == 0] == 0.0)
    # inside mask should remain unchanged (equal to original)
    assert np.all(out1[mask1 == 1] == img1[mask1 == 1])

    out2 = out.channels["488"][1]
    assert np.sum(out2) == 50.0


def test_remove_background_single_plane_and_nuclei_mask_option():
    img = make_image()
    mask = make_mask()

    data = DummyData(channels={"561": img.copy()}, segmentation=None, nuclei_segmentation=mask)

    t = RemoveBackgroundTransform(channel_keys=["561"], background_padding=1, mask="nuclei")
    out = t(data)

    out_img = out.channels["561"][0]
    assert np.all(out_img[mask == 1] == img[mask == 1])


def test_remove_background_raises_when_no_masks_available():
    img = make_image()
    data = DummyData(channels={"488": [img.copy()]}, segmentation=None, nuclei_segmentation=None)

    t = RemoveBackgroundTransform(channel_keys=["488"], background_padding=0, mask="cell")
    with pytest.raises(ValueError):
        t(data)


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
        assert 0 <= result.channels["405"][0].min() <= result.channels["405"][0].max() <= 1

    def test_full_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline."""
        transforms = [
            SelectPlanesTransform(plane_selection="middle"),
            NormalizeTransform(method="minmax", channel_keys=["bf", "405"]),
            GaussianFilterTransform(sigma=1.0, channel_keys=["bf", "405"]),
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

        assert np.sum(result.segmentation - cell_data.segmentation[0]) == 0

        # Should be normalized [0, 1]
        assert result.channels["405"][0].min() >= 0
        assert result.channels["405"][0].max() <= 1

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
        cell_data = CellData(cell_id="test", channels=channels, segmentation=[np.zeros((250, 250))])

        result = pipeline(cell_data)

        # Should return unchanged data
        assert result.channels["405"][0][0, 0] == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
