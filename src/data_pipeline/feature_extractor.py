"""Feature extraction module for cell microscopy data.

This module provides tools for extracting morphological and intensity features
from cell images and segmentation masks.
"""

import numpy as np
import traceback
from typing import Dict
from skimage.measure import label, regionprops


class FeatureExtractor:
    """Class for feature extraction — separated from data loading.

    Provides methods to extract morphological and intensity-based features
    from microscopy images and segmentation masks.
    """

    def extract_morphological_features(
        self, image: np.ndarray, mask: np.ndarray, cell_name: str, type: str
    ) -> Dict:
        """Extract morphological features from a labeled region.

        Args:
            image: intensity image used to compute intensity-based props
            mask: binary mask containing the region of interest
            cell_name: identifier for logging/errors
            type: prefix used for returned feature names (e.g., 'cell' or 'nucleus')
        Returns:
            A dict of morphological feature values.
        """
        mask[mask > 0] = 1.0  # ensure binary mask
        labeled_mask = label(mask)
        properties = regionprops(labeled_mask, intensity_image=image)

        if not properties:
            raise ValueError(f"No regions found in mask for cell {cell_name}.")

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
        self, image: np.ndarray, mask_nucleus: np.ndarray, mask_cell: np.ndarray
    ) -> Dict:
        """Extract simple intensity summary features.

        Computes total intensity inside the nucleus and total intensity in the
        cytoplasm (cell mask minus nucleus mask).
        """
        # total intensity inside nucleus
        total_intensity_nucleus = np.sum(image[mask_nucleus > 0])

        # cytoplasm = cell mask AND not nucleus
        cytoplasm_mask = (mask_cell > 0) & (mask_nucleus == 0)
        total_intensity_outside_nucleus = np.sum(image[cytoplasm_mask])

        features = {
            "total_intensity_nucleus": total_intensity_nucleus,
            "total_intensity_outside_nucleus": total_intensity_outside_nucleus,
        }

        return features

    def extract_all_features(self, cell_data) -> Dict:
        """Extract all features for a single cell data dict.

        The input `cell_data` is expected to be a CellData object with channels
        ('405', '561', '488') and segmentation masks (segmentation, nuclei_segmentation).
        """
        features = {}
        try:
            middle_idx = len(cell_data.channels["405"]) // 2

            nucleus_image = cell_data.channels["405"][middle_idx]
            seg_mask = cell_data.segmentation[middle_idx]
            nuclei_seg_mask = cell_data.nuclei_segmentation[middle_idx]
            cell_name = cell_data.metadata.get("cell_id", None)
            # extract morphological features
            morphological_features_cell = self.extract_morphological_features(
                nucleus_image, seg_mask, cell_name, type="cell"
            )
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

            # Intensity features: note the correct ordering of args (nucleus, cell mask)
            intensity_features = self.extract_intensity_features(
                nucleus_image, nuclei_seg_mask, seg_mask
            )
            features.update(intensity_features)

        except Exception:
            print(f"Error extracting features for {cell_name}:")
            traceback.print_exc()

        return features
