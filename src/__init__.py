"""
Iris ML Pipeline Package
A comprehensive Machine Learning project for cell cycle analysis
and H5 cell image data processing.
"""

__version__ = "1.0.0"
__author__ = "Iris ML Team"
__email__ = "team@iris-ml.com"

# Import main classes for easy access
try:
    from .data_processing import (
        H5DataCleaner,
        H5CellDataset,
        CellImageDataset,
        CellImageTransform,
        FUCCIRepresentationTransform,
        FeatureExtractor,
        CellDataModule,
    )

    PYTORCH_AVAILABLE = True

except ImportError as e:
    # Fallback if dependencies are missing
    print(f"Warning: Not all modules could be imported: {e}")
    H5DataCleaner = None
    H5CellDataset = None
    CellImageDataset = None
    CellImageTransform = None
    FUCCIRepresentationTransform = None
    FeatureExtractor = None
    CellDataModule = None
    PYTORCH_AVAILABLE = False

# Define what is available on import
__all__ = [
    "H5DataCleaner",
    "H5CellDataset",
    "CellImageDataset",
    "CellImageTransform",
    "FUCCIRepresentationTransform",
    "FeatureExtractor",
    "CellDataModule",
    "PYTORCH_AVAILABLE",
]

# Füge PyTorch-Klassen hinzu falls verfügbar
if PYTORCH_AVAILABLE:
    __all__.extend(["H5CellDataset", "PyTorchH5DataLoader", "PyTorchMLDataPipeline"])

# Paket-Metadaten
__pkg_info__ = {
    "name": "iris-ml-pipeline",
    "version": __version__,
    "description": "Machine Learning Pipeline für Iris und H5-Zellbild-Daten",
    "pytorch_available": PYTORCH_AVAILABLE,
    "supported_formats": ["csv", "parquet", "h5", "hdf5"],
    "ml_algorithms": ["RandomForest", "SVM", "LogisticRegression", "KNN"],
    "image_processing": ["gaussian_filter", "normalization", "illumination_correction"],
}
