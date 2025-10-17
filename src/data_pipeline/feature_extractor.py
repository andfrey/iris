"""Feature extraction module for cell microscopy data.

This module provides tools for extracting morphological and intensity features
from cell images and segmentation masks.
"""

import numpy as np
import traceback
from typing import Dict
from skimage.measure import label, regionprops


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

            nucleus_image = cell_data["405"][middle_idx]
            seg_mask = cell_data["seg"][middle_idx]
            red_image = cell_data["561"][middle_idx]
            green_image = cell_data["488"][middle_idx]
            nuclei_seg_mask = cell_data["nuclei_seg"][middle_idx]
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

        except Exception as e:
            print(f"Fehler beim Extrahieren der Features für {cell_name}:")
            traceback.print_exc()

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
