"""
Test suite for ModularCellDataModule using configuration files.
"""

import sys
import tempfile
from pathlib import Path

import pytest
import torch
import yaml
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.dataset import ModularCellDataModule

# Skip all tests in this file when run by pre-commit
pytestmark = pytest.mark.skipif(
    os.getenv("PRECOMMIT") == "1", reason="Skipping data module tests during pre-commit"
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def data_config_path():
    """Return path to the default data config."""
    return Path(__file__).resolve().parent / ".." / "configs/data_config.yaml"


@pytest.fixture
def datamodule(data_config_path):
    """Create a data module instance."""
    return ModularCellDataModule(data_config_path=data_config_path)


@pytest.fixture
def setup_datamodule(datamodule):
    """Setup data module for training."""
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    return datamodule


# ============================================================================
# Test Config Loading
# ============================================================================


def test_config_loading(data_config_path):
    """Test that config file is loaded correctly."""
    dm = ModularCellDataModule(data_config_path=data_config_path)

    # Check that config is loaded
    assert dm.config is not None
    assert isinstance(dm.config, dict)

    # Check key parameters are set
    assert dm.h5_path is not None
    assert dm.batch_size > 0
    assert dm.num_workers >= 0
    assert len(dm.data_split) == 3
    assert sum(dm.data_split) == pytest.approx(1.0)


def test_config_hyperparameters_saved(datamodule):
    """Test that config is saved to hyperparameters."""
    assert hasattr(datamodule, "hparams")
    assert len(datamodule.hparams) > 0


# ============================================================================
# Test Initialization
# ============================================================================


def test_datamodule_initialization(data_config_path):
    """Test that data module initializes correctly."""
    dm = ModularCellDataModule(data_config_path=data_config_path)

    # Check attributes
    assert dm.h5_path is not None
    assert dm.batch_size > 0
    assert dm.num_workers >= 0
    assert dm.use_quality_filters is not None

    # Check datasets are not yet initialized
    assert dm.train_dataset is None
    assert dm.val_dataset is None
    assert dm.full_dataset is None


# ============================================================================
# Test Setup and Splitting
# ============================================================================


def test_datamodule_setup(datamodule):
    """Test that setup creates datasets correctly."""
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    # Check datasets are created
    assert datamodule.full_dataset is not None
    assert datamodule.train_dataset is not None
    assert datamodule.val_dataset is not None
    assert datamodule.test_dataset is not None

    # Check lengths
    full_len = len(datamodule.full_dataset)
    train_len = len(datamodule.train_dataset)
    val_len = len(datamodule.val_dataset)
    test_len = len(datamodule.test_dataset)

    assert train_len + val_len + test_len == full_len
    print(
        f"Dataset sizes - Train: {train_len}, Val: {val_len}, Test: {test_len}, Total: {full_len}"
    )


def test_datamodule_split_ratios(setup_datamodule):
    """Test that split ratios are approximately correct."""
    dm = setup_datamodule

    full_len = len(dm.full_dataset)
    train_len = len(dm.train_dataset)
    val_len = len(dm.val_dataset)
    test_len = len(dm.test_dataset)

    expected_train = dm.data_split[0]
    expected_val = dm.data_split[1]
    expected_test = dm.data_split[2]

    # Allow for rounding differences
    assert abs(train_len / full_len - expected_train) < 0.05
    assert abs(val_len / full_len - expected_val) < 0.05
    assert abs(test_len / full_len - expected_test) < 0.05


def test_datamodule_reproducible_split():
    """Test that splits are reproducible with same seed."""
    config_path = "configs/data_config.yaml"

    # Create two data modules with same seed
    dm1 = ModularCellDataModule(data_config_path=config_path)
    dm1.prepare_data()
    dm1.setup(stage="fit")

    dm2 = ModularCellDataModule(data_config_path=config_path)
    dm2.prepare_data()
    dm2.setup(stage="fit")

    # Get first samples from train datasets
    sample1_img, sample1_label = dm1.train_dataset[0]
    sample2_img, sample2_label = dm2.train_dataset[0]

    # Should be identical with same seed
    assert torch.allclose(sample1_img, sample2_img)
    assert torch.allclose(sample1_label, sample2_label)


# ============================================================================
# Test DataLoaders
# ============================================================================


@pytest.mark.skipif(os.getenv("PRECOMMIT") == "1", reason="Skip in pre-commit")
@pytest.mark.slow
def test_train_dataloader(setup_datamodule):
    """Test train dataloader creation."""
    dm = setup_datamodule
    train_loader = dm.train_dataloader()

    assert train_loader is not None
    assert len(train_loader) > 0
    assert train_loader.batch_size == dm.batch_size


@pytest.mark.skipif(os.getenv("PRECOMMIT") == "1", reason="Skip in pre-commit")
@pytest.mark.slow
def test_val_dataloader(setup_datamodule):
    """Test validation dataloader creation."""
    dm = setup_datamodule
    val_loader = dm.val_dataloader()

    assert val_loader is not None
    assert len(val_loader) > 0
    assert val_loader.batch_size == dm.batch_size


@pytest.mark.skipif(os.getenv("PRECOMMIT") == "1", reason="Skip in pre-commit")
@pytest.mark.slow
def test_test_dataloader(setup_datamodule):
    """Test test dataloader creation."""
    dm = setup_datamodule
    test_loader = dm.test_dataloader()

    assert test_loader is not None
    assert len(test_loader) > 0
    assert test_loader.batch_size == dm.batch_size


@pytest.mark.skipif(os.getenv("PRECOMMIT") == "1", reason="Skip in pre-commit")
@pytest.mark.slow
def test_predict_dataloader(setup_datamodule):
    """Test prediction dataloader creation."""
    dm = setup_datamodule
    predict_loader = dm.predict_dataloader()

    assert predict_loader is not None
    assert len(predict_loader) > 0
    assert predict_loader.batch_size == dm.batch_size


# ============================================================================
# Test Batch Shapes and Types
# ============================================================================


@pytest.mark.skipif(os.getenv("PRECOMMIT") == "1", reason="Skip in pre-commit")
@pytest.mark.slow
def test_batch_shape(setup_datamodule):
    """Test that batches have correct shape."""
    dm = setup_datamodule
    train_loader = dm.train_dataloader()

    # Get first batch
    images, labels = next(iter(train_loader))

    # Check shapes
    batch_size = images.shape[0]
    assert batch_size <= dm.batch_size
    assert images.ndim == 4  # (B, C, H, W)
    assert labels.ndim == 2  # (B, 2)
    assert labels.shape[1] == 2  # [488, 561] intensities

    print(f"Batch shape - Images: {images.shape}, Labels: {labels.shape}")


@pytest.mark.skipif(os.getenv("PRECOMMIT") == "1", reason="Skip in pre-commit")
@pytest.mark.slow
def test_batch_types(setup_datamodule):
    """Test that batch elements have correct types."""
    dm = setup_datamodule
    train_loader = dm.train_dataloader()

    images, labels = next(iter(train_loader))

    # Check types
    assert isinstance(images, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert images.dtype == torch.float32
    assert labels.dtype == torch.float32


@pytest.mark.skipif(os.getenv("PRECOMMIT") == "1", reason="Skip in pre-commit")
@pytest.mark.slow
def test_batch_values(setup_datamodule):
    """Test that batch values are in reasonable ranges."""
    dm = setup_datamodule
    train_loader = dm.train_dataloader()

    images, labels = next(iter(train_loader))

    # Images should be normalized (approximately)
    # After standardization, values should be roughly in [-3, 3]
    assert not torch.isnan(images).any()
    assert not torch.isinf(images).any()

    # Labels should be log intensities (positive or zero)
    assert not torch.isnan(labels).any()
    assert not torch.isinf(labels).any()
    assert (labels >= 0).all() or (labels <= 15).all()  # Reasonable log intensity range

    print(
        f"Image stats - Min: {images.min():.2f}, Max: {images.max():.2f}, Mean: {images.mean():.2f}"
    )
    print(
        f"Label stats - Min: {labels.min():.2f}, Max: {labels.max():.2f}, Mean: {labels.mean():.2f}"
    )


@pytest.mark.skipif(os.getenv("PRECOMMIT") == "1", reason="Skip in pre-commit")
@pytest.mark.slow
def test_channel_count(setup_datamodule):
    """Test that images have correct number of channels."""
    dm = setup_datamodule
    train_loader = dm.train_dataloader()

    images, _ = next(iter(train_loader))

    # Should have 6 channels (2 channels × 3 planes)
    assert images.shape[1] == 6
    print(f"Channel count: {images.shape[1]}")


# ============================================================================
# Test Integration
# ============================================================================


@pytest.mark.skipif(os.getenv("PRECOMMIT") == "1", reason="Skip in pre-commit")
@pytest.mark.slow
def test_full_training_loop_iteration(setup_datamodule):
    """Test that we can iterate through full training loop."""
    dm = setup_datamodule
    train_loader = dm.train_dataloader()

    batch_count = 0
    total_samples = 0

    for images, labels in train_loader:
        batch_count += 1
        total_samples += images.shape[0]

        # Basic checks
        assert images.shape[0] == labels.shape[0]
        assert not torch.isnan(images).any()
        assert not torch.isnan(labels).any()

    assert batch_count > 0
    assert total_samples == len(dm.train_dataset)

    print(f"Training loop - Batches: {batch_count}, Total samples: {total_samples}")


@pytest.mark.skipif(os.getenv("PRECOMMIT") == "1", reason="Skip in pre-commit")
@pytest.mark.slow
def test_validation_loop_iteration(setup_datamodule):
    """Test that we can iterate through validation loop."""
    dm = setup_datamodule
    val_loader = dm.val_dataloader()

    batch_count = 0
    for images, labels in val_loader:
        batch_count += 1
        assert images.shape[0] == labels.shape[0]

    assert batch_count > 0
    print(f"Validation loop - Batches: {batch_count}")


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
