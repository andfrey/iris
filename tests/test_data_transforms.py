"""
Tests for data_transforms module.
"""

from copy import deepcopy
import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.data_transforms import (
    SelectPlanesTransform,
    NormalizeTransform,
    TransformPipeline,
    RemoveBackgroundTransform,
    CenterCellTransform,
)


class TestCenterCellTransform:
    def test_features_unchanged_after_centering(self):
        """Test that features from FeatureExtractor are unchanged after centering."""
        from src.data_pipeline.feature_extractor import FeatureExtractor

        # Create a simple cell mask and nucleus mask
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:7, 2:8] = 1
        nucleus_mask = np.zeros((10, 10), dtype=np.uint8)
        nucleus_mask[4:6, 4:8] = 1

        # Channel with cell in same location
        channel = np.zeros((10, 10), dtype=np.float32)
        channel[2:7, 2:7] = 5.0

        channels = {"405": [channel]}

        # CellData expects segmentation and nuclei_segmentation
        class DummyCellData:
            def __init__(self):
                self.cell_id = "test"
                self.channels = channels
                self.segmentation = [mask]
                self.nuclei_segmentation = [nucleus_mask]
                self.metadata = {"cell_id": "test"}

        cell_data = DummyCellData()
        extractor = FeatureExtractor()
        features_before = extractor.extract_all_features(cell_data)

        # Apply centering
        transform = CenterCellTransform(dimension=10)
        cell_data_centered = transform(cell_data)
        features_after = extractor.extract_all_features(cell_data_centered)

        # Features should be identical
        assert features_before == features_after

    """Test CenterCellTransform."""

    def test_center_cell_basic(self):
        """Test centering a cell in the image."""
        transform = CenterCellTransform(dimension=10)

        # Create a mask with a cell in the top-left corner
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[1:5, 1:4] = 1

        # Channel with cell in same location
        channel = np.zeros((10, 10), dtype=np.float32)
        channel[1:5, 1:4] = 5.0

        channels = {"488": [channel]}
        cell_data = CellData(cell_id="test", channels=channels, segmentation=[mask])

        result = transform(cell_data)

        # The cell should now be centered in the output
        centered_channel = result.channels["488"][0]
        # Find where the cell is now
        cell_indices = np.argwhere(centered_channel == 5.0)
        # Should be centered around the middle
        assert np.all((cell_indices >= 3) & (cell_indices <= 6))

    def test_center_cell_empty_mask(self):
        """Test centering with an empty mask (should not move cell)."""
        transform = CenterCellTransform(dimension=10)

        mask = np.zeros((10, 10), dtype=np.uint8)
        channel = np.ones((10, 10), dtype=np.float32)
        channels = {"488": [channel]}
        cell_data = CellData(cell_id="test", channels=channels, segmentation=[mask])

        result = transform(cell_data)
        # Should remain unchanged since mask is empty
        assert np.allclose(result.channels["488"][0], channel)


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


def test_remove_background_mean_method():
    """Test background removal using background_mean method."""
    # Create an image with known values
    img = np.full((10, 12), 100.0, dtype=float)
    # Set some pixels inside the mask to higher values
    img[2:8, 3:9] = 200.0

    # Create a mask
    mask = make_mask(shape=(10, 12), on_region=(slice(2, 8), slice(3, 9)))

    data = DummyData(channels={"488": [img.copy()]}, segmentation=[mask])

    # Apply background removal with background_mean method
    t = RemoveBackgroundTransform(
        channel_keys=["488"], method="background_mean", background_padding=0, mask="cell"
    )
    out = t(data)

    out_img = out.channels["488"][0]

    # The mean outside of masked region (all 100.0) is 100.0
    # So the entire image should be img - 100.0
    expected = img - 100.0

    assert np.allclose(out_img, expected), f"Expected {expected}, but got {out_img}"


def test_remove_background_mean_method_multiplane():
    """Test background_mean method with multiple planes."""
    # Create two planes with different values
    img1 = np.full((10, 12), 50.0, dtype=float)
    img1[2:8, 3:9] = 150.0  # Mean inside mask = 150.0

    img2 = np.full((10, 12), 30.0, dtype=float)
    img2[2:8, 3:9] = 80.0  # Mean inside mask = 80.0

    mask1 = make_mask(shape=(10, 12), on_region=(slice(2, 8), slice(3, 9)))
    mask2 = make_mask(shape=(10, 12), on_region=(slice(2, 8), slice(3, 9)))

    data = DummyData(channels={"561": [img1.copy(), img2.copy()]}, segmentation=[mask1, mask2])

    t = RemoveBackgroundTransform(
        channel_keys=["561"], method="background_mean", background_padding=0, mask="cell"
    )
    out = t(data)

    # First plane: subtract mean outside masked region (50.0)
    out1 = out.channels["561"][0]
    expected1 = img1 - 50.0
    assert np.allclose(out1, expected1)

    # Second plane: subtract mean outside masked region (30.0)
    out2 = out.channels["561"][1]
    expected2 = img2 - 30.0
    assert np.allclose(out2, expected2)


class TestTransformPipeline:
    """Test TransformPipeline."""

    def test_pipeline_execution_order(self):
        """Test transforms are applied in order."""
        transforms = [
            SelectPlanesTransform(plane_selection="middle"),
            NormalizeTransform(method="minmax", channel_keys=["405"]),
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
        assert len(result.channels["405"]) == 1
        assert 0 <= result.channels["405"][0].min() <= result.channels["405"][0].max() <= 1

    def test_full_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline."""
        transforms = [
            SelectPlanesTransform(plane_selection="middle"),
            NormalizeTransform(method="minmax", channel_keys=["bf", "405"]),
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
