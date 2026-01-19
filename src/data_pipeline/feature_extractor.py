"""Feature extraction module for cell microscopy data.

This module provides tools for extracting morphological and intensity features
from cell images and segmentation masks.
"""

import pandas as pd
import mahotas
import numpy as np
import traceback
from typing import Dict, Union, Optional, List
from skimage.measure import label, regionprops
import re


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
        roundness = (4 * np.pi * prop.area) / (prop.perimeter**2) if prop.perimeter > 0 else 0
        return {
            f"{type}_area": prop.area,
            f"{type}_perimeter": prop.perimeter,
            f"{type}_mean_intensity": prop.mean_intensity,
            f"{type}_eccentricity": prop.eccentricity,
            f"{type}_solidity": prop.solidity,
            f"{type}_extent": prop.extent,
            f"{type}_major_axis_length": prop.major_axis_length,
            f"{type}_minor_axis_length": prop.minor_axis_length,
            f"{type}_roundness": roundness,
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

    def extract_texture_features(self, image: np.ndarray, mask: np.ndarray, type: str) -> Dict:
        """Extract texture features from the image within the mask."""
        props = regionprops(mask)

        # a. Extract the bounding box of the specific cell (Crop)
        # Coordinates from regionprops are (min_row, min_col, max_row, max_col)
        minr, minc, maxr, maxc = props[0].bbox

        cell_crop = image[minr:maxr, minc:maxc]
        mask_crop = mask[minr:maxr, minc:maxc]

        # Mask the crop so we only analyze the cell/nucleus, not the background in the box
        cell_isolated = np.where(mask_crop == 0, cell_crop, 0)

        # Rescale intensity for Haralick
        # Haralick requires integer inputs and a small range (e.g., 0-255 or 0-63).
        # If we use raw 16-bit (0-65535), the GLCM matrix is huge and computation fails.
        # Here we convert to 8-bit (0-255).
        if cell_isolated.max() > 0:
            scale_factor = 255 / cell_isolated.max()
            cell_8bit = (cell_isolated * scale_factor).astype(np.uint8)
        else:
            cell_8bit = cell_isolated.astype(np.uint8)
        texture_feats = mahotas.features.haralick(cell_8bit)
        mean_texture = texture_feats.mean(axis=0)

        # Append specific features we care about:
        # Index 1 = Contrast (Edges/Mitosis)
        # Index 8 = Entropy (Disorder)
        return {
            f"haralick_contrast_{type}": mean_texture[1],
            f"haralick_entropy_{type}": mean_texture[8],
            f"haralick_correlation_{type}": mean_texture[2],
        }

    def polynomial_transform(self, features: dict):
        """Apply polynomial transformations to selected features"""
        transformed_features = {}
        for key, value in features.items():
            if isinstance(value, (int, float)):
                transformed_features[f"{key}^2"] = value**2
                transformed_features[f"{key}^3"] = value**3

        return transformed_features

    def extract_all_features(self, cell_data) -> Dict:
        """Extract all features for a single cell data dict.

        The input `cell_data` is expected to be a CellData object with channels
        ('405', '561', '488') and segmentation masks (segmentation, nuclei_segmentation).
        """
        features = {}
        try:
            middle_idx = len(cell_data.segmentation) // 2 if len(cell_data.segmentation) > 0 else 0

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

            texture_features_cell = self.extract_texture_features(
                nucleus_image, seg_mask, type="cell"
            )
            texture_features_nucleus = self.extract_texture_features(
                nucleus_image, nuclei_seg_mask, type="nucleus"
            )

            features.update(morphological_features_cell)
            features.update(morphological_features_nucleus)
            features.update(texture_features_cell)
            features.update(texture_features_nucleus)

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


def polynomial_transform(
    features: Union[dict, pd.DataFrame, pd.Series], columns: Optional[List] = None, degree: int = 3
):
    if columns is None:
        if isinstance(features, dict):
            columns = list(features.keys())
        elif isinstance(features, (pd.DataFrame, pd.Series)):
            columns = features.columns
        else:
            raise TypeError(
                "Unsupported type for features. Expected dict, pd.DataFrame, or pd.Series."
            )

    for column in columns:
        if re.search(r"\^\d+$", column):
            if int(column.split("^")[-1]) > degree:
                del features[column]
            continue
        for d in range(2, degree + 1):
            poly_column = str(column) + "^" + str(d)
            try:
                features[poly_column] = features[column] ** d
            except:
                pass
    return features
