"""
Tests for compute_fucci_labels function and related dataset functionality.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.dataset import compute_fucci_labels, ModularCellDataset
from src.data_pipeline.data_sources import CellData


class TestComputeFucciLabels:
    """Test suite for compute_fucci_labels function."""

    def create_mock_cell_data(
        self,
        intensity_488: float = 100.0,
        intensity_561: float = 200.0,
        num_planes: int = 3,
        image_size: tuple = (250, 250),
        background_noise: float = 10.0,
        mask_fraction: float = 0.1,
    ) -> CellData:
        """
        Create mock cell data with known intensities for testing.

        Args:
            intensity_488: Mean intensity for 488 channel inside mask
            intensity_561: Mean intensity for 561 channel inside mask
            num_planes: Number of z-planes
            image_size: Size of each image plane (H, W)
            background_noise: Background intensity outside mask
            mask_fraction: Fraction of image covered by mask
        """
        channels = {}
        segmentation = []

        # Create mask (circular region in center)
        H, W = image_size
        center_y, center_x = H // 2, W // 2
        radius = int(np.sqrt(mask_fraction * H * W / np.pi))

        y, x = np.ogrid[:H, :W]
        mask_single = ((y - center_y) ** 2 + (x - center_x) ** 2 <= radius**2).astype(
            np.uint8
        )

        # Create channels with known intensities
        for channel, intensity in [("488", intensity_488), ("561", intensity_561)]:
            planes = []
            for _ in range(num_planes):
                # Create image: background + signal in mask
                plane = np.ones(image_size, dtype=np.float32) * background_noise
                plane[mask_single > 0] = intensity + background_noise
                # Add small random noise
                plane += np.random.randn(*image_size) * 0.5
                planes.append(plane)
            channels[channel] = planes

        # Create segmentation masks
        for _ in range(num_planes):
            segmentation.append(mask_single.copy())

        # Add other required channels (optional)
        channels["405"] = [
            np.zeros(image_size, dtype=np.float32) for _ in range(num_planes)
        ]
        channels["bf"] = [
            np.ones(image_size, dtype=np.float32) * 50 for _ in range(num_planes)
        ]

        return CellData(
            cell_id="test_cell",
            channels=channels,
            segmentation=segmentation,
            nuclei_segmentation=[mask_single.copy() for _ in range(num_planes)],
        )

    def test_basic_functionality(self):
        """Test basic functionality with known intensities."""
        # Create cell with known intensities
        cell_data = self.create_mock_cell_data(
            intensity_488=100.0, intensity_561=200.0, background_noise=10.0
        )

        # Compute labels
        labels = compute_fucci_labels(cell_data)

        # Check output shape
        assert labels.shape == (2,), f"Expected shape (2,), got {labels.shape}"
        assert labels.dtype == np.float32, f"Expected dtype float32, got {labels.dtype}"

        # Check that values are log-transformed (should be positive)
        assert labels[0] > 0, "488 label should be positive (log-transformed)"
        assert labels[1] > 0, "561 label should be positive (log-transformed)"

        # Check relative ordering (561 should be higher)
        # After background subtraction: 488 has ~90, 561 has ~190
        # log(190) > log(90)
        assert labels[1] > labels[0], "561 intensity should be higher than 488"

    def test_background_subtraction(self):
        """Test that background is properly subtracted."""
        # Create cell with same signal but different backgrounds
        cell_low_bg = self.create_mock_cell_data(
            intensity_488=100.0, intensity_561=100.0, background_noise=5.0
        )

        cell_high_bg = self.create_mock_cell_data(
            intensity_488=100.0, intensity_561=100.0, background_noise=50.0
        )

        labels_low = compute_fucci_labels(cell_low_bg)
        labels_high = compute_fucci_labels(cell_high_bg)

        # After background subtraction, both should give similar results
        # Allow some tolerance due to noise
        np.testing.assert_allclose(
            labels_low,
            labels_high,
            rtol=0.1,
            atol=0.2,
            err_msg="Background subtraction should normalize intensities",
        )

    def test_multiple_planes_averaging(self):
        """Test that multiple planes are properly averaged."""
        # Create cell with 3 planes
        cell_3planes = self.create_mock_cell_data(
            intensity_488=100.0, intensity_561=200.0, num_planes=3
        )

        # Create cell with 5 planes (same intensities)
        cell_5planes = self.create_mock_cell_data(
            intensity_488=100.0, intensity_561=200.0, num_planes=5
        )

        labels_3 = compute_fucci_labels(cell_3planes)
        labels_5 = compute_fucci_labels(cell_5planes)

        # Should give similar results regardless of plane count
        np.testing.assert_allclose(
            labels_3,
            labels_5,
            rtol=0.05,
            atol=0.1,
            err_msg="Averaging should work across different plane counts",
        )

    def test_missing_segmentation(self):
        """Test error handling when segmentation is missing."""
        # Create cell data without segmentation
        channels = {
            "488": [np.ones((250, 250), dtype=np.float32) * 100],
            "561": [np.ones((250, 250), dtype=np.float32) * 200],
        }

        cell_data = CellData(
            cell_id="test_cell",
            channels=channels,
            segmentation=None,
            nuclei_segmentation=None,
        )

        # Should raise ValueError
        with pytest.raises(ValueError, match="Missing segmentation mask"):
            compute_fucci_labels(cell_data)

    def test_missing_channel(self):
        """Test error handling when a required channel is missing."""
        # Create cell data without 561 channel
        channels = {
            "488": [np.ones((250, 250), dtype=np.float32) * 100],
        }
        segmentation = [np.ones((250, 250), dtype=np.uint8)]

        cell_data = CellData(
            cell_id="test_cell",
            channels=channels,
            segmentation=segmentation,
            nuclei_segmentation=None,
        )

        # Should raise ValueError for missing channel
        with pytest.raises(ValueError):
            compute_fucci_labels(cell_data)

    def test_plane_mask_mismatch(self):
        """Test error handling when plane and mask counts don't match."""
        # Create cell with mismatched planes and masks
        channels = {
            "488": [np.ones((250, 250), dtype=np.float32) * 100 for _ in range(3)],
            "561": [np.ones((250, 250), dtype=np.float32) * 200 for _ in range(3)],
        }
        segmentation = [
            np.ones((250, 250), dtype=np.uint8) for _ in range(2)
        ]  # Only 2 masks!

        cell_data = CellData(
            cell_id="test_cell",
            channels=channels,
            segmentation=segmentation,
            nuclei_segmentation=None,
        )

        # Should raise ValueError
        with pytest.raises(
            ValueError, match="planes and segmentation planes count mismatch"
        ):
            compute_fucci_labels(cell_data)

    def test_empty_mask(self):
        """Test handling of empty segmentation mask."""
        # Create cell with empty mask
        channels = {
            "488": [np.ones((250, 250), dtype=np.float32) * 100 for _ in range(3)],
            "561": [np.ones((250, 250), dtype=np.float32) * 200 for _ in range(3)],
        }
        segmentation = [
            np.zeros((250, 250), dtype=np.uint8) for _ in range(3)
        ]  # Empty masks!

        cell_data = CellData(
            cell_id="test_cell",
            channels=channels,
            segmentation=segmentation,
            nuclei_segmentation=None,
        )

        with pytest.raises(ValueError):
            compute_fucci_labels(cell_data)

    def test_deterministic_output(self):
        """Test that the function gives consistent output."""
        cell_data = self.create_mock_cell_data(intensity_488=100.0, intensity_561=200.0)

        # Compute labels multiple times
        labels1 = compute_fucci_labels(cell_data)
        labels2 = compute_fucci_labels(cell_data)
        labels3 = compute_fucci_labels(cell_data)

        # Should be identical
        np.testing.assert_array_equal(labels1, labels2)
        np.testing.assert_array_equal(labels2, labels3)

    def test_different_mask_sizes(self):
        """Test with different mask sizes (different cell sizes)."""
        # Small mask (small cell)
        cell_small = self.create_mock_cell_data(
            intensity_488=100.0, intensity_561=200.0, mask_fraction=0.05  # 5% of image
        )

        # Large mask (large cell)
        cell_large = self.create_mock_cell_data(
            intensity_488=100.0, intensity_561=200.0, mask_fraction=0.2  # 20% of image
        )

        labels_small = compute_fucci_labels(cell_small)
        labels_large = compute_fucci_labels(cell_large)

        # Should give similar results (intensity is averaged, not summed)
        np.testing.assert_allclose(
            labels_small,
            labels_large,
            rtol=0.1,
            atol=0.2,
            err_msg="Mask size should not affect mean intensity",
        )

    def test_high_background_noise(self):
        """Test behavior with high background noise."""
        # Very high background
        cell_data = self.create_mock_cell_data(
            intensity_488=150.0,
            intensity_561=250.0,
            background_noise=100.0,  # High background
        )

        labels = compute_fucci_labels(cell_data)

        # After background subtraction: 488=50, 561=150
        # Should still work
        assert labels[0] > 0, "Should handle high background"
        assert labels[1] > labels[0], "Relative ordering should be preserved"

    def test_clipping_negative_values(self):
        """Test that negative values after background subtraction are clipped."""
        # Create cell where signal is lower than background in some areas
        channels = {
            "488": [np.ones((250, 250), dtype=np.float32) * 50],  # Low signal
            "561": [np.ones((250, 250), dtype=np.float32) * 100],
        }

        # Mask in center
        mask = np.zeros((250, 250), dtype=np.uint8)
        mask[100:150, 100:150] = 1

        # High background outside mask
        channels["488"][0][mask == 0] = 200.0  # Much higher than signal!
        channels["561"][0][mask == 0] = 200.0

        segmentation = [mask]

        cell_data = CellData(
            cell_id="test_cell",
            channels=channels,
            segmentation=segmentation,
            nuclei_segmentation=None,
        )

        # Should handle negative values (clip to 0)
        labels = compute_fucci_labels(cell_data)

        # Should not crash and return finite values
        assert np.isfinite(
            labels
        ).all(), "Should return finite values even with negative signals"


class TestModularCellDataset:
    """Test suite for ModularCellDataset integration."""

    def test_dataset_with_fucci_labels(self, tmp_path):
        """Test that dataset correctly uses compute_fucci_labels."""
        # This is an integration test that would need a real H5 file
        # For now, we'll create a simple mock
        pytest.skip("Integration test requires H5 file - implement with fixtures")


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_plane(self):
        """Test with single z-plane."""
        channels = {
            "488": [np.ones((250, 250), dtype=np.float32) * 110],
            "561": [np.ones((250, 250), dtype=np.float32) * 210],
        }
        mask = np.ones((250, 250), dtype=np.uint8)
        mask[:100, :100] = 0  # Some background

        segmentation = [mask]

        cell_data = CellData(
            cell_id="test_cell",
            channels=channels,
            segmentation=segmentation,
            nuclei_segmentation=None,
        )

        labels = compute_fucci_labels(cell_data)
        assert labels.shape == (2,), "Should work with single plane"

    def test_very_small_intensities(self):
        """Test with very small intensity values."""
        channels = {
            "488": [np.ones((250, 250), dtype=np.float32) * 0.1],
            "561": [np.ones((250, 250), dtype=np.float32) * 0.2],
        }
        mask = np.ones((250, 250), dtype=np.uint8)
        segmentation = [mask]

        cell_data = CellData(
            cell_id="test_cell",
            channels=channels,
            segmentation=segmentation,
            nuclei_segmentation=None,
        )

        labels = compute_fucci_labels(cell_data)

        # Log of very small numbers will be negative
        assert np.isfinite(labels).all(), "Should handle very small intensities"

    def test_very_large_intensities(self):
        """Test with very large intensity values."""
        channels = {
            "488": [np.ones((250, 250), dtype=np.float32) * 10000],
            "561": [np.ones((250, 250), dtype=np.float32) * 20000],
        }
        mask = np.ones((250, 250), dtype=np.uint8)
        segmentation = [mask]

        cell_data = CellData(
            cell_id="test_cell",
            channels=channels,
            segmentation=segmentation,
            nuclei_segmentation=None,
        )

        labels = compute_fucci_labels(cell_data)
        assert np.isfinite(labels).all(), "Should handle very large intensities"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
