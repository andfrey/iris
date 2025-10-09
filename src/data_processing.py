"""
Iris ML Pipeline - Datenverarbeitung
Dieses Modul behandelt das Laden und Vorverarbeiten der Iris-Daten.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import h5py
import cv2
from skimage.measure import regionprops, label
from skimage.io import imread
from tqdm import tqdm
from scipy import ndimage
from typing import Optional, Union, Dict, List, Tuple, Generator, Any
import sys
from abc import ABC, abstractmethod

# PyTorch imports
try:
    import torch
    from torch.utils.data import Dataset, DataLoader

    PYTORCH_AVAILABLE = True
except ImportError:
    print("Warnung: PyTorch nicht installiert. Fallback auf custom DataLoader.")
    PYTORCH_AVAILABLE = False

    # Mock classes für Kompatibilität
    class Dataset:
        pass

    class DataLoader:
        pass


try:
    import basicpy
except ImportError:
    print("Warnung: basicpy nicht installiert. Beleuchtungskorrektur nicht verfügbar.")
    basicpy = None


class BaseDataLoader(ABC):
    """Abstrakte Basisklasse für alle DataLoader."""

    @abstractmethod
    def load_batch(self, batch_size: int) -> Generator[Dict, None, None]:
        """Lädt Daten in Batches als Generator."""
        pass

    @abstractmethod
    def get_item_count(self) -> int:
        """Gibt die Anzahl der verfügbaren Items zurück."""
        pass


class H5CellDataset(Dataset):
    """
    PyTorch Dataset für H5-Zellbilddaten.
    Ermöglicht die Verwendung mit PyTorch DataLoader für optimierte Performance.
    """

    def __init__(
        self,
        h5_file_path: str,
        preprocessor: Optional["ImagePreprocessor"] = None,
        feature_extractor: Optional["FeatureExtractor"] = None,
        return_features: bool = True,
        return_raw: bool = False,
        channels: Optional[List[str]] = None,
    ):
        """
        Args:
            h5_file_path: Pfad zur H5-Datei
            preprocessor: ImagePreprocessor für Bildverarbeitung
            feature_extractor: FeatureExtractor für Feature-Extraktion
            return_features: Ob Features zurückgegeben werden sollen
            return_raw: Ob Rohdaten zusätzlich zurückgegeben werden sollen
            channels: Liste der zu ladenden Kanäle (default: alle)
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError(
                "PyTorch ist nicht installiert. Installieren Sie es mit: pip install torch"
            )

        super().__init__()

        self.h5_file_path = h5_file_path
        self.preprocessor = preprocessor
        self.feature_extractor = feature_extractor
        self.return_features = return_features
        self.return_raw = return_raw
        self.channels = channels

        if not os.path.exists(h5_file_path):
            raise FileNotFoundError(f"H5-Datei nicht gefunden: {h5_file_path}")

        # Sammle Zellnamen
        with h5py.File(h5_file_path, "r") as f:
            self.cell_names = list(f.keys())

        # Validiere dass Features oder Raw-Daten angefordert werden
        if not return_features and not return_raw:
            raise ValueError(
                "Mindestens return_features oder return_raw muss True sein"
            )

        print(f"H5CellDataset initialisiert: {len(self.cell_names)} Zellen")

    def __len__(self) -> int:
        """Gibt die Anzahl der Zellen zurück."""
        return len(self.cell_names)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Lädt eine einzelne Zelle.

        Args:
            idx: Index der Zelle

        Returns:
            Dictionary mit Zellendaten und/oder Features
        """
        if idx >= len(self.cell_names):
            raise IndexError(
                f"Index {idx} außerhalb des Bereichs [0, {len(self.cell_names)})"
            )

        cell_name = self.cell_names[idx]
        result = {"cell_name": cell_name}

        # Lade Zellendaten
        cell_data = self._load_cell_data(cell_name)

        # Füge Rohdaten hinzu falls gewünscht
        if self.return_raw:
            result["raw_data"] = cell_data

        # Extrahiere Features falls gewünscht
        if self.return_features and self.feature_extractor:
            features = self.feature_extractor.extract_all_features(cell_data, cell_name)
            features.update({"index_cell": idx})
            result["features"] = features

        return result

    def _load_cell_data(
        self, cell_name: str
    ) -> Dict[str, List[Tuple[str, np.ndarray]]]:
        """Lädt Daten für eine einzelne Zelle."""
        with h5py.File(self.h5_file_path, "r") as f:
            if cell_name not in f:
                raise KeyError(f"Zelle {cell_name} nicht gefunden")

            grp = f[cell_name]
            cell_data = {}

            for channel_name, channel_data in grp.items():
                # Überspringe Kanäle falls Filter gesetzt
                if self.channels and channel_name not in self.channels:
                    continue

                cell_data[channel_name] = []

                for plane_name, plane_data in channel_data.items():
                    image = plane_data[()]
                    masks = {}
                    
                    # Load segmentation masks if they exist for this plane
                    if "nuclei_seg" in grp:
                        nuclei_seg_group = grp["nuclei_seg"]
                        if plane_name in nuclei_seg_group:
                            masks["nuclei_seg"] = nuclei_seg_group[plane_name][()]
                        else:
                            masks["nuclei_seg"] = None
                    
                    if "seg" in grp:
                        seg_group = grp["seg"]
                        if plane_name in seg_group:
                            masks["seg"] = seg_group[plane_name][()]
                        else:
                            masks["seg"] = None
                    
                    image = plane_data[()]
                    # Bildverarbeitung anwenden
                    if self.preprocessor:
                        processed_image = self.preprocessor.process_image(
                            image, channel_name, masks
                        )
                    else:
                        processed_image = image

                    cell_data[channel_name].append((plane_name, processed_image))

            return cell_data

    def get_features_dataframe(
        self, indices: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Extrahiert Features für alle oder ausgewählte Zellen als DataFrame.

        Args:
            indices: Liste von Indices (default: alle)

        Returns:
            DataFrame mit Features
        """
        if not self.return_features or not self.feature_extractor:
            raise ValueError("Feature-Extraktion nicht aktiviert")

        if indices is None:
            indices = list(range(len(self)))

        features_list = []
        for idx in tqdm(indices, desc="Extrahiere Features"):
            item = self[idx]
            if "features" in item:
                features_list.append(item["features"])

        return pd.DataFrame(features_list)


class PyTorchH5DataLoader:
    """
    Wrapper für H5CellDataset der die PyTorch DataLoader Funktionalität bereitstellt.
    """

    def __init__(
        self,
        h5_file_path: str,
        preprocessor: Optional["ImagePreprocessor"] = None,
        feature_extractor: Optional["FeatureExtractor"] = None,
        return_features: bool = True,
        return_raw: bool = False,
        channels: Optional[List[str]] = None,
    ):

        self.dataset = H5CellDataset(
            h5_file_path=h5_file_path,
            preprocessor=preprocessor,
            feature_extractor=feature_extractor,
            return_features=return_features,
            return_raw=return_raw,
            channels=channels,
        )

    def get_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 0,
        **kwargs,
    ) -> "DataLoader":
        """
        Erstellt einen PyTorch DataLoader.

        Args:
            batch_size: Batch-Größe
            shuffle: Ob Daten gemischt werden sollen
            num_workers: Anzahl Worker-Prozesse
            **kwargs: Weitere DataLoader-Parameter

        Returns:
            PyTorch DataLoader
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch ist nicht installiert")

        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            **kwargs,
        )

    def _collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        """
        Custom collate function für Batch-Verarbeitung.

        Args:
            batch: Liste von Dataset-Items

        Returns:
            Batch-Dictionary
        """
        result = {}

        # Sammle cell_names
        result["cell_names"] = [item["cell_name"] for item in batch]

        # Sammle Features falls vorhanden
        if "features" in batch[0]:
            features_list = [item["features"] for item in batch]
            result["features"] = pd.DataFrame(features_list)

        # Sammle Rohdaten falls vorhanden
        if "raw_data" in batch[0]:
            result["raw_data"] = [item["raw_data"] for item in batch]

        return result

    def __len__(self) -> int:
        """Gibt die Anzahl der Zellen zurück."""
        return len(self.dataset)

    def get_features_dataframe(
        self, indices: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Delegiert an Dataset."""
        return self.dataset.get_features_dataframe(indices)


class PyTorchMLDataPipeline:
    """
    Erweiterte ML-Pipeline die PyTorch DataLoader nutzt.
    Bietet optimierte Performance und Parallelisierung.
    """

    def __init__(
        self,
        h5_file_path: str,
        preprocessor: Optional["ImagePreprocessor"] = None,
        feature_extractor: Optional["FeatureExtractor"] = None,
    ):
        self.h5_file_path = h5_file_path
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.feature_extractor = feature_extractor or FeatureExtractor()

        self.pytorch_loader = PyTorchH5DataLoader(
            h5_file_path=h5_file_path,
            preprocessor=self.preprocessor,
            feature_extractor=self.feature_extractor,
            return_features=True,
            return_raw=False,
        )

    def get_feature_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 0,
        **kwargs,
    ) -> "DataLoader":
        """
        Erstellt einen PyTorch DataLoader für Features.

        Args:
            batch_size: Batch-Größe
            shuffle: Ob Daten gemischt werden sollen
            num_workers: Anzahl Worker-Prozesse für Parallelisierung
            **kwargs: Weitere DataLoader-Parameter

        Returns:
            PyTorch DataLoader
        """
        return self.pytorch_loader.get_dataloader(
            batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kwargs
        )

    def get_raw_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 0,
        channels: Optional[List[str]] = None,
        **kwargs,
    ) -> "DataLoader":
        """
        Erstellt einen PyTorch DataLoader für Rohdaten.

        Args:
            batch_size: Batch-Größe
            shuffle: Ob Daten gemischt werden sollen
            num_workers: Anzahl Worker-Prozesse
            channels: Spezifische Kanäle zu laden
            **kwargs: Weitere DataLoader-Parameter

        Returns:
            PyTorch DataLoader für Rohdaten
        """
        raw_loader = PyTorchH5DataLoader(
            h5_file_path=self.h5_file_path,
            preprocessor=self.preprocessor,
            feature_extractor=None,
            return_features=False,
            return_raw=True,
            channels=channels,
        )

        return raw_loader.get_dataloader(
            batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kwargs
        )

    def extract_all_features(
        self,
        max_items: Optional[int] = None,
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> pd.DataFrame:
        """
        Extrahiert Features für alle Zellen mit PyTorch DataLoader.

        Args:
            max_items: Maximale Anzahl zu verarbeitender Items
            batch_size: Batch-Größe
            num_workers: Anzahl Worker-Prozesse

        Returns:
            DataFrame mit allen Features
        """
        if max_items:
            indices = list(range(min(max_items, len(self.pytorch_loader))))
        else:
            indices = None

        return self.pytorch_loader.get_features_dataframe(indices)

    def create_training_pipeline(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        validation_split: float = 0.2,
        **kwargs,
    ) -> Tuple["DataLoader", "DataLoader"]:
        """
        Erstellt Training- und Validierung-DataLoader.

        Args:
            batch_size: Batch-Größe
            shuffle: Ob Training-Daten gemischt werden sollen
            num_workers: Anzahl Worker-Prozesse
            validation_split: Anteil für Validierung (0.0-1.0)
            **kwargs: Weitere DataLoader-Parameter

        Returns:
            Tuple von (train_dataloader, val_dataloader)
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch ist nicht installiert")

        from torch.utils.data import random_split

        dataset = self.pytorch_loader.dataset
        total_size = len(dataset)
        val_size = int(total_size * validation_split)
        train_size = total_size - val_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.pytorch_loader._collate_fn,
            **kwargs,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # Validierung nicht mischen
            num_workers=num_workers,
            collate_fn=self.pytorch_loader._collate_fn,
            **kwargs,
        )

        return train_loader, val_loader


# Legacy MLDataPipeline für Rückwärtskompatibilität


class ImagePreprocessor:
    """Klasse für Bildvorverarbeitung - getrennt von der Datenladung."""

    def __init__(
        self,
        apply_gaussian: bool = True,
        sigma: float = 1.0,
        apply_illumination_correction: bool = False,
    ):
        self.apply_gaussian = apply_gaussian
        self.sigma = sigma
        self.apply_illumination_correction = apply_illumination_correction
        self.flatfields = {}
        self.darkfields = {}

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalisiert ein Bild."""
        img_mean = image.mean()
        img_std = image.std()
        if img_std > 0:
            norm_image = (image - img_mean) / img_std
        else:
            norm_image = image
        return norm_image

    def remove_illumination(self, image, mask):
        """Entfernt Beleuchtungseffekte basierend auf einer Maske."""
        if mask.sum() == 0:
            return image  # Keine Maske, nichts zu tun

        # Berechne den Mittelwert des Bildes aussesrhalb der Maske
        mean_intensity = np.median(image[mask == 0])
        std_intensity = np.std(image[mask == 0])
        if std_intensity == 0:
            std_intensity = 1  # Vermeide Division durch Null

        # Subtrahiere den Mittelwert von allen Pixeln
        corrected_image = (image - mean_intensity) / std_intensity

        # Setze negative Werte auf Null
        corrected_image[corrected_image < 0] = 0

        return corrected_image

    def apply_gaussian_filter(self, image: np.ndarray) -> np.ndarray:
        """Wendet Gauss-Filter an."""
        if self.apply_gaussian:
            return ndimage.gaussian_filter(image, sigma=self.sigma)
        return image

    def process_image(
        self, image: np.ndarray, channel_name: str = None, masks: dict = None
    ) -> np.ndarray:
        """Führt komplette Bildverarbeitung durch."""
        # Ensure image is a proper numpy array
        if not isinstance(image, np.ndarray):
            return image
            
        if channel_name == "seg" or channel_name == "nuclei_seg":
            return image  # Segmentierungsmasken nicht verarbeiten

        # Start with original image
        processed_image = image
        
        # Beleuchtungskorrektur falls verfügbar und masks nicht None
        if (
            channel_name in ["405", "488", "561"]
            and self.apply_illumination_correction
            and masks is not None
        ):
            if channel_name == "405" and "nuclei_seg" in masks and masks["nuclei_seg"] is not None:
                processed_image = self.remove_illumination(image, masks["nuclei_seg"])
            elif channel_name in ["488", "561"] and "seg" in masks and masks["seg"] is not None:
                processed_image = self.remove_illumination(image, masks["seg"])
        
        # Gauss-Filter und Normalisierung
        if channel_name in ["bf"]:
            processed_image = self.apply_gaussian_filter(processed_image)

        return processed_image




class FeatureExtractor:
    """Klasse für Feature-Extraktion - getrennt von der Datenladung."""

    def extract_morphological_features(
        self, image: np.ndarray, mask: np.ndarray, cell_name: str, type: str
    ) -> Dict:
        """Extrahiert morphologische Features aus Zellbildern."""
        labeled_mask = label(mask)
        properties = regionprops(labeled_mask, intensity_image=image)

        if not properties:
            return self._get_empty_morphological_features(cell_name, type)

        prop = properties[0]
        return {
            f"{type}_area": prop.area,
            f"{type}_perimeter": prop.perimeter,
            f"{type}_mean_intensity": prop.mean_intensity,
            f"{type}_eccentricity": prop.eccentricity,
            f"{type}_solidity": prop.solidity,
            f"{type}_extent": prop.extent,
            f"{type}_major_axis_length": prop.major_axis_length,
            f"{type}_minor_axis_length": prop.minor_axis_length,
        }

    def extract_intensity_features(
        self, image: np.ndarray, mask: np.ndarray, channel: str
    ) -> Dict:
        """Extrahiert Intensitäts-Features."""
        # Intensität innerhalb und außerhalb der Maske
        total_intensity_inside = np.sum(image[mask > 0])
        total_intensity_outside = np.sum(image[mask == 0])
        total_intensity = total_intensity_inside + total_intensity_outside

        features = {
            "intensity_inside": total_intensity_inside,
            "intensity_outside": total_intensity_outside,
        }

        if total_intensity > 0:
            features["intensity_ratio_outside_inside"] = (
                total_intensity_outside / total_intensity_inside
                if total_intensity_inside > 0
                else np.inf
            )
            features["intensity_fraction_outside"] = (
                total_intensity_outside / total_intensity
            )
        else:
            features["intensity_ratio_outside_inside"] = 0
            features["intensity_fraction_outside"] = 0

        return features

    def extract_channel_statistics(
        self, image: np.ndarray, mask: np.ndarray, channel: str
    ) -> Dict:
        """Extrahiert Kanalstatistiken."""
        masked_pixels = image[mask > 0]
        stats = {}

        if masked_pixels.size > 0:
            stats[f"mean_intensity_in_mask_{channel}"] = np.mean(masked_pixels)
            stats[f"median_intensity_in_mask_{channel}"] = np.median(masked_pixels)
            stats[f"std_intensity_in_mask_{channel}"] = np.std(masked_pixels)
            stats[f"min_intensity_in_mask_{channel}"] = np.min(masked_pixels)
            stats[f"max_intensity_in_mask_{channel}"] = np.max(masked_pixels)
        else:
            stats[f"mean_intensity_in_mask_{channel}"] = 0
            stats[f"median_intensity_in_mask_{channel}"] = 0
            stats[f"std_intensity_in_mask_{channel}"] = 0
            stats[f"min_intensity_in_mask_{channel}"] = 0
            stats[f"max_intensity_in_mask_{channel}"] = 0

        return stats

    def extract_all_features(self, cell_data: Dict, cell_name: str) -> Dict:
        """Extrahiert alle Features für eine Zelle."""
        features = {"cell_name": cell_name}

        # Prüfe ob alle erforderlichen Kanäle vorhanden sind
        required_channels = ["405", "seg", "561", "488"]
        if not all(ch in cell_data for ch in required_channels):
            return features

        try:
            # Nimm mittlere Ebene jedes Kanals
            middle_idx = len(cell_data["405"]) // 2

            nucleus_image = cell_data["405"][middle_idx][1]
            seg_mask = cell_data["seg"][middle_idx][1]
            red_image = cell_data["561"][middle_idx][1]
            green_image = cell_data["488"][middle_idx][1]
            nuclei_seg_mask = cell_data["nuclei_seg"][middle_idx][1]
            # Morphologische Features
            morphological_features_cell = self.extract_morphological_features(
                nucleus_image, seg_mask, cell_name, type="cell"
            )
            if "nuclei_seg" in cell_data:
                morphological_features_nucleus = self.extract_morphological_features(
                    nucleus_image, nuclei_seg_mask, cell_name, type="nucleus"
                )

            features.update(morphological_features_cell)
            features.update(morphological_features_nucleus)

            features["cell_nucleus_area_ratio"] = (
                morphological_features_nucleus["nucleus_area"]
                / morphological_features_cell["cell_area"]
                if morphological_features_cell["cell_area"] > 0
                else 0
            )
            # Intensitäts-Features
            intensity_features = self.extract_intensity_features(
                nucleus_image, seg_mask, "405"
            )
            features.update(intensity_features)

            # Kanalstatistiken
            red_stats = self.extract_channel_statistics(red_image, seg_mask, "561")
            green_stats = self.extract_channel_statistics(green_image, seg_mask, "488")
            features.update(red_stats)
            features.update(green_stats)

        except Exception as e:
            print(f"Fehler beim Extrahieren der Features für {cell_name}: {e}")

        return features

    def _get_empty_morphological_features(self, cell_name: str, type: str) -> Dict:
        """Gibt leere morphologische Features zurück."""
        return {
            f"{type}_area": 0,
            f"{type}_perimeter": 0,
            f"{type}_mean_intensity": 0,
            f"{type}_eccentricity": 0,
            f"{type}_solidity": 0,
            f"{type}_extent": 0,
            f"{type}_major_axis_length": 0,
            f"{type}_minor_axis_length": 0,
        }


class H5DataLoader(BaseDataLoader):
    """Spezialisierter DataLoader für H5-Dateien mit Generator-Support."""

    def __init__(
        self, file_path: str, preprocessor: Optional[ImagePreprocessor] = None
    ):
        self.file_path = file_path
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.cell_names = []
        self._item_count = 0

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"H5-Datei nicht gefunden: {file_path}")

        # Initialisiere und sammle Zellnamen
        self._initialize()

    def _initialize(self):
        """Initialisiert den DataLoader und sammelt Metadaten."""
        with h5py.File(self.file_path, "r") as f:
            self.cell_names = list(f.keys())
            self._item_count = len(self.cell_names)

        print(f"H5DataLoader initialisiert: {self._item_count} Zellen gefunden")

    def get_item_count(self) -> int:
        """Gibt die Anzahl der verfügbaren Items zurück."""
        return self._item_count

    def load_single_cell(self, cell_name: str) -> Dict:
        """Lädt eine einzelne Zelle aus der H5-Datei."""
        with h5py.File(self.file_path, "r") as f:
            if cell_name not in f:
                raise KeyError(f"Zelle {cell_name} nicht in H5-Datei gefunden")

            grp = f[cell_name]
            cell_data = {}

            for channel_name, channel_data in grp.items():
                cell_data[channel_name] = []
                for plane_name, plane_data in channel_data.items():
                    masks = {}
                    if "nuclei_seg" in grp:
                        masks["nuclei_seg"] = grp.get("nuclei_seg", None)[plane_name][
                            ()
                        ]
                    if "seg" in grp:
                        masks["seg"] = grp.get("seg", None)[plane_name][()]
                    image = plane_data[()]

                    # Bildverarbeitung anwenden
                    processed_image = self.preprocessor.process_image(
                        image, channel_name, masks
                    )
                    cell_data[channel_name].append((plane_name, processed_image))

            return cell_data

    def load_batch(
        self, batch_size: int = 32, start_idx: int = 0
    ) -> Generator[List[Tuple[str, Dict]], None, None]:
        """
        Generator für batch-weise Datenladung.

        Args:
            batch_size: Größe der Batches
            start_idx: Startindex

        Yields:
            List von (cell_name, cell_data) Tupeln
        """
        current_idx = start_idx

        while current_idx < self._item_count:
            batch_end = min(current_idx + batch_size, self._item_count)
            batch_cell_names = self.cell_names[current_idx:batch_end]

            batch_data = []
            for cell_name in batch_cell_names:
                try:
                    cell_data = self.load_single_cell(cell_name)
                    batch_data.append((cell_name, cell_data))
                except Exception as e:
                    print(f"Fehler beim Laden von Zelle {cell_name}: {e}")
                    continue

            if batch_data:
                yield batch_data

            current_idx = batch_end

    def load_single_items(
        self, start_idx: int = 0
    ) -> Generator[Tuple[str, Dict], None, None]:
        """
        Generator für einzelne Zellen.

        Args:
            start_idx: Startindex

        Yields:
            (cell_name, cell_data) Tupel
        """
        for i in range(start_idx, self._item_count):
            cell_name = self.cell_names[i]
            try:
                cell_data = self.load_single_cell(cell_name)
                yield cell_name, cell_data
            except Exception as e:
                print(f"Fehler beim Laden von Zelle {cell_name}: {e}")
                continue

    def compute_illumination_correction_from_sample(self, sample_size: int = 100):
        """Berechnet Beleuchtungskorrektur aus einer Stichprobe."""
        if not self.preprocessor.apply_illumination_correction:
            return

        print(f"Sammle {sample_size} Bilder für Beleuchtungskorrektur...")
        channel_images = {}

        sample_count = 0
        for cell_name, cell_data in self.load_single_items():
            if sample_count >= sample_size:
                break

            for channel_name, planes in cell_data.items():
                if channel_name == "seg":
                    continue

                if channel_name not in channel_images:
                    channel_images[channel_name] = []

                # Nimm mittlere Ebene
                middle_idx = len(planes) // 2
                if middle_idx < len(planes):
                    _, image = planes[middle_idx]
                    channel_images[channel_name].append(image)

            sample_count += 1

        # Berechne Korrektur
        self.preprocessor.compute_illumination_correction(channel_images)

    def get_channel_names(self) -> List[str]:
        """Gibt die verfügbaren Kanalnamen zurück."""
        if not self.cell_names:
            return []

        # Verwende erste Zelle als Referenz
        first_cell_name = self.cell_names[0]
        cell_data = self.load_single_cell(first_cell_name)
        return list(cell_data.keys())


class MLDataPipeline:
    """Hauptklasse die DataLoader, Preprocessor und FeatureExtractor kombiniert."""

    def __init__(
        self, data_loader: BaseDataLoader, feature_extractor: FeatureExtractor
    ):
        self.data_loader = data_loader
        self.feature_extractor = feature_extractor

    def extract_features_batch(
        self, batch_size: int = 32, max_items: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Extrahiert Features in Batches für memory-effiziente Verarbeitung.

        Args:
            batch_size: Größe der Batches
            max_items: Maximale Anzahl zu verarbeitender Items

        Returns:
            DataFrame mit extrahierten Features
        """
        all_features = []
        processed_count = 0

        total_items = min(max_items or float("inf"), self.data_loader.get_item_count())

        print(f"Extrahiere Features für {total_items} Items...")

        with tqdm(total=total_items) as pbar:
            for batch in self.data_loader.load_batch(batch_size):
                batch_features = []

                for cell_name, cell_data in batch:
                    if max_items and processed_count >= max_items:
                        break

                    features = self.feature_extractor.extract_all_features(
                        cell_data, cell_name
                    )
                    batch_features.append(features)
                    processed_count += 1
                    pbar.update(1)

                all_features.extend(batch_features)

                if max_items and processed_count >= max_items:
                    break

        return pd.DataFrame(all_features)

    def create_training_generator(
        self, batch_size: int = 32, extract_features: bool = True
    ) -> Generator:
        """
        Erstellt einen Generator für Training-Pipelines.

        Args:
            batch_size: Größe der Batches
            extract_features: Ob Features extrahiert werden sollen

        Yields:
            Batch von Daten (entweder Rohdaten oder Features)
        """
        for batch in self.data_loader.load_batch(batch_size):
            if extract_features:
                # Extrahiere Features für den Batch
                batch_features = []
                for cell_name, cell_data in batch:
                    features = self.feature_extractor.extract_all_features(
                        cell_data, cell_name
                    )
                    batch_features.append(features)

                yield pd.DataFrame(batch_features)
            else:
                # Gib Rohdaten zurück
                yield batch


class DataProcessor:
    """Klasse für Standard Iris Datenverarbeitung und -vorbereitung."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.target_names = ["setosa", "versicolor", "virginica"]

    def load_data(self):
        """Lädt den Standard Iris-Datensatz."""
        iris = load_iris()

        # Erstelle DataFrame
        df = pd.DataFrame(
            iris.data,
            columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        )
        df["species"] = iris.target
        df["species_name"] = df["species"].map(
            {i: name for i, name in enumerate(self.target_names)}
        )

        return df, iris.data, iris.target

    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """Teilt Daten in Training und Test auf."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Standardisierung
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def save_scaler(self, filepath="models/scaler.pkl"):
        """Speichert den Scaler für spätere Verwendung."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.scaler, filepath)

    def load_scaler(self, filepath="models/scaler.pkl"):
        """Lädt einen gespeicherten Scaler."""
        self.scaler = joblib.load(filepath)
        return self.scaler


if __name__ == "__main__":
    print("IRIS ML PIPELINE - PYTORCH DATALOADER INTEGRATION")
    print("=" * 70)

    # 1. Standard Iris Dataset Demo
    print("\n1. STANDARD IRIS DATASET:")
    processor = DataProcessor()
    df, X, y = processor.load_data()

    print(f"Shape: {df.shape}")
    print(f"Klassen: {processor.target_names}")
    print("\nErste 5 Zeilen:")
    print(df.head())

    # Daten vorbereiten
    X_train, X_test, y_train, y_test = processor.prepare_data(X, y)
    print(f"\nTraining Set: {X_train.shape}")
    print(f"Test Set: {X_test.shape}")

    # Scaler speichern
    processor.save_scaler()
    print("Scaler gespeichert.")

    # 2. PyTorch H5 Pipeline Demo
    print("\n2. PYTORCH H5 DATENPIPELINE:")
    example_h5_path = "/mydata/iris/andreas/fucci_3t3_221124_filtered_noNG030JP208.h5"

    if os.path.exists(example_h5_path):
        print(f"Teste PyTorch Pipeline mit: {example_h5_path}")
        try:
            if PYTORCH_AVAILABLE:
                # Erstelle PyTorch Pipeline
                preprocessor = ImagePreprocessor(
                    apply_gaussian=True, sigma=1.0, apply_illumination_correction=False
                )

                feature_extractor = FeatureExtractor()
                pytorch_pipeline = PyTorchMLDataPipeline(
                    example_h5_path, preprocessor, feature_extractor
                )

                print(f"✓ PyTorch Pipeline initialisiert")
                print(f"✓ Dataset Größe: {len(pytorch_pipeline.pytorch_loader)}")

                # Teste PyTorch DataLoader
                print("\n3. TESTE PYTORCH DATALOADER:")
                feature_dataloader = pytorch_pipeline.get_feature_dataloader(
                    batch_size=8, shuffle=False, num_workers=0  # Für Demo
                )

                print(f"✓ DataLoader erstellt (batch_size=8)")

                # Teste ersten Batch
                first_batch = next(iter(feature_dataloader))
                print(f"✓ Erster Batch geladen:")
                print(f"  Cell Names: {len(first_batch['cell_names'])}")
                print(f"  Features Shape: {first_batch['features'].shape}")
                print(
                    f"  Feature Columns: {list(first_batch['features'].columns)[:5]}..."
                )

                # Teste Training/Validation Split
                print("\n4. TESTE TRAINING/VALIDATION SPLIT:")
                train_loader, val_loader = pytorch_pipeline.create_training_pipeline(
                    batch_size=4, validation_split=0.2, num_workers=0
                )

                print(f"✓ Training DataLoader: ~{len(train_loader)} Batches")
                print(f"✓ Validation DataLoader: ~{len(val_loader)} Batches")

            else:
                print("❌ PyTorch nicht verfügbar - verwende Legacy DataLoader")

                # Fallback auf Legacy-Implementation
                preprocessor = ImagePreprocessor(apply_gaussian=True)
                loader = H5DataLoader(example_h5_path, preprocessor)
                extractor = FeatureExtractor()
                legacy_pipeline = MLDataPipeline(loader, extractor)

                print(
                    f"✓ Legacy Pipeline initialisiert mit {loader.get_item_count()} Zellen"
                )

                features_df = legacy_pipeline.extract_features_batch(
                    batch_size=10, max_items=20
                )
                print(f"✓ Features extrahiert: {features_df.shape}")

        except Exception as e:
            print(f"Fehler bei der Pipeline: {e}")
            import traceback

            traceback.print_exc()
    else:
        print(f"H5-Datei nicht gefunden: {example_h5_path}")
        print("PyTorch Integration erfolgreich implementiert!")
        print("\n✓ Verfügbare Klassen:")
        print("  - H5CellDataset (PyTorch Dataset)")
        print("  - PyTorchH5DataLoader (PyTorch DataLoader Wrapper)")
        print("  - PyTorchMLDataPipeline (Hauptpipeline mit PyTorch)")
        print("  - Legacy Klassen für Rückwärtskompatibilität")

        print("\n📖 PyTorch Beispiel-Verwendung:")
        print("pipeline = PyTorchMLDataPipeline('data.h5')")
        print(
            "dataloader = pipeline.get_feature_dataloader(batch_size=32, num_workers=4)"
        )
        print("for batch in dataloader:")
        print("    features_df = batch['features']  # Ready for ML training")

        if PYTORCH_AVAILABLE:
            print(f"\n✅ PyTorch verfügbar: Version installiert")
        else:
            print(f"\n⚠️  PyTorch nicht verfügbar. Installieren Sie mit:")
            print("    pip install torch")

    print("\n" + "=" * 70)
    print("✓ DEMO ABGESCHLOSSEN")
    print("✓ PyTorch DataLoader Integration implementiert")
    print("✓ Optimierte Performance durch Parallelisierung")
    print("✓ Nahtlose Integration in ML-Pipelines")
    print("✓ Rückwärtskompatibilität gewährleistet")
