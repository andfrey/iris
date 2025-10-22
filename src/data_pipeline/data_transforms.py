"""
Transform pipeline for data preprocessing.
Composable transforms that can be chained together.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import cv2
import numpy as np
from scipy import ndimage
from copy import deepcopy
from skimage.restoration import rolling_ball
from skimage.morphology import ball, disk


class Transform(ABC):
    """Base class for all transforms"""

    @abstractmethod
    def __call__(self, data):
        """Apply transform to cell data"""
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get transform configuration for reproducibility"""
        pass


class ChannelTransform(Transform):
    """Base class for transforms that operate on individual channels"""

    def __init__(self, channel_keys: Optional[List[str]] = None):
        self.channel_keys = channel_keys

    def __call__(self, data):
        data = deepcopy(data)
        channels_to_transform = self.channel_keys or list(data.channels.keys())

        for key in channels_to_transform:
            if key in data.channels:
                planes = data.channels[key]
                # Transform each plane
                if isinstance(planes, list):
                    data.channels[key] = [self.transform_image(plane, key) for plane in planes]
                else:
                    data.channels[key] = [self.transform_image(planes, key)]

        return data

    @abstractmethod
    def transform_image(self, image: np.ndarray, channel_key: str) -> np.ndarray:
        """Transform a single 2D image"""
        pass


class RemoveBackgroundTransform(Transform):
    """Remove background using morphological opening"""

    def __init__(
        self,
        channel_keys: Optional[List[str]] = None,
        background_padding: int = 15,
        mask: str = "cell",
    ):
        self.channel_keys = channel_keys
        self.background_padding = background_padding
        self.mask = mask  # 'cell' or 'nuclei'
        if mask not in ["cell", "nuclei"]:
            raise ValueError("mask must be 'cell' or 'nuclei'")

    def __call__(self, data):
        data = deepcopy(data)
        channels_to_transform = self.channel_keys or list(data.channels.keys())

        for key in channels_to_transform:
            if key in data.channels:
                planes = data.channels[key]
                masks = data.segmentation if self.mask == "cell" else data.nuclei_segmentation
                if masks is None:
                    raise ValueError(
                        f"No {self.mask} segmentation mask available for background removal"
                    )
                # Transform each plane
                if isinstance(planes, list):
                    data.channels[key] = self._remove_background(planes, masks)
                else:
                    data.channels[key] = self._remove_background([planes], [masks])

        return data

    def _remove_background(
        self, images: List[np.ndarray], masks: List[np.ndarray]
    ) -> List[np.ndarray]:
        # Subtract background
        for image, mask in zip(images, masks):
            if np.sum(mask) < 10:
                mask = masks[
                    len(masks) // 2
                ]  # Use middle plane if no mask provided has less than 10 pixels
                if mask is None:
                    raise ValueError("No segmentation mask available for background removal")
            mask_height, mask_width = mask.shape
            # Pad mask
            if self.background_padding > 0:
                mask = cv2.resize(
                    mask,
                    (
                        mask_width + self.background_padding * 2,
                        mask_height + self.background_padding * 2,
                    ),
                    interpolation=cv2.INTER_CUBIC,
                )
                mask = mask[
                    self.background_padding : -self.background_padding,
                    self.background_padding : -self.background_padding,
                ]
            image[mask == 0] = 0.0

        return images

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "RemoveBackgroundTransform",
            "channel_keys": self.channel_keys,
            "background_padding": self.background_padding,
            "mask": self.mask,
        }


class FUCCIScaleTransform(ChannelTransform):
    """Scale FUCCI channel intensities based on predefined factors"""

    def __init__(
        self,
        channel_keys: Optional[List[str]] = ["488", "561"],
        scale_divider_488: float = 18_000,
        scale_divider_561: float = 40_000,
    ):
        super().__init__(channel_keys)
        self.scale_factors = {
            "488": 1.0 / scale_divider_488,
            "561": 1.0 / scale_divider_561,
        }

    def transform_image(self, image: np.ndarray, channel_key: str) -> np.ndarray:
        scale_factor = self.scale_factors.get(channel_key)
        if scale_factor is None:
            raise ValueError(f"No scale factor defined for channel {channel_key}")

        return image * scale_factor

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "FUCCIScaleTransform",
            "channel_keys": self.channel_keys,
            "scale_factors": self.scale_factors,
        }


class CropTransform(Transform):
    """Crop image to bounding box of segmentation mask"""

    def __init__(self, padding: int = 0, dimension: int = 250):
        """
        Args:
            padding: Additional pixels to include around the mask bounding box
        """
        self.padding = padding
        self.dimension = dimension

    def __call__(self, data):
        data = deepcopy(data)

        # Get the segmentation mask to use
        masks = data.segmentation

        if not isinstance(masks, list):
            raise TypeError("Expected list of planes for segmentation mask")
        rmins = []
        rmaxs = []
        cmins = []
        cmaxs = []
        for plane_mask in masks:
            # If mask is empty, skip
            if np.sum(plane_mask) < 10:
                continue
            # Find bounding box of non-zero regions
            rows = np.any(plane_mask, axis=1)
            cols = np.any(plane_mask, axis=0)

            if not rows.any() or not cols.any():
                # Empty mask, return original data
                return data

            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]

            crop_dim = max(rmax - rmin, cmax - cmin)

            padding_r = (crop_dim - (rmax - rmin)) // 2 + self.padding
            padding_c = (crop_dim - (cmax - cmin)) // 2 + self.padding

            # Add padding
            h, w = plane_mask.shape
            rmins.append(max(0, rmin - padding_r))
            rmaxs.append(min(h, rmax + padding_r + 1))
            cmins.append(max(0, cmin - padding_c))
            cmaxs.append(min(w, cmax + padding_c + 1))

        rmin = min(rmins)
        rmax = max(rmaxs)
        cmin = min(cmins)
        cmax = max(cmaxs)

        # Crop all channels
        for key in data.channels:
            planes = data.channels[key]
            if isinstance(planes, list):
                data.channels[key] = [
                    self._fit_to_dimension(plane[rmin:rmax, cmin:cmax]) for plane in planes
                ]
            else:
                data.channels[key] = [self._fit_to_dimension(planes[rmin:rmax, cmin:cmax])]

        # Crop segmentation masks
        if data.segmentation is not None:
            if isinstance(data.segmentation, list):
                data.segmentation = [
                    self._fit_to_dimension(plane[rmin:rmax, cmin:cmax])
                    for plane in data.segmentation
                ]
            else:
                data.segmentation = self._fit_to_dimension(data.segmentation[rmin:rmax, cmin:cmax])

        if data.nuclei_segmentation is not None:
            if isinstance(data.nuclei_segmentation, list):
                data.nuclei_segmentation = [
                    self._fit_to_dimension(plane[rmin:rmax, cmin:cmax])
                    for plane in data.nuclei_segmentation
                ]
            else:
                data.nuclei_segmentation = self._fit_to_dimension(
                    data.nuclei_segmentation[rmin:rmax, cmin:cmax]
                )

        return data

    def _fit_to_dimension(self, image: np.ndarray) -> np.ndarray:
        """Resize or pad image to target dimension"""
        image_dim = image.shape[0]
        target_dim = self.dimension

        # Crop if larger than target
        if image_dim > target_dim:
            image = cv2.resize(image, (target_dim, target_dim), interpolation=cv2.INTER_AREA)
        else:
            image = cv2.resize(image, (target_dim, target_dim), interpolation=cv2.INTER_CUBIC)

        # Pad if smaller than target
        h, w = image.shape
        pad_h = max(0, target_dim - h)
        pad_w = max(0, target_dim - w)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        if pad_h > 0 or pad_w > 0:
            image = np.pad(
                image,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
                constant_values=0,
            )

        return image

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "CropTransform",
            "padding": self.padding,
            "dimension": self.dimension,
        }


class NormalizeTransform(ChannelTransform):
    """Normalize channel values"""

    def __init__(
        self,
        channel_keys: Optional[List[str]] = None,
        method: str = "minmax",  # 'minmax', 'percentile'
    ):
        super().__init__(channel_keys)
        self.method = method

    def transform_image(self, image: np.ndarray, channel_key: str) -> np.ndarray:
        if self.method == "minmax":
            min_val = image.min()
            max_val = image.max()
            if max_val - min_val > 0:
                normalized = (image - min_val) / (max_val - min_val)
            else:
                normalized = image / max_val

        elif self.method == "standardize":
            mean = image.mean()
            std = image.std()
            if std > 0:
                normalized = (image - mean) / std
            else:
                normalized = image - mean

        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

        return normalized

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "NormalizeTransform",
            "channel_keys": self.channel_keys,
            "method": self.method,
        }


class GaussianFilterTransform(ChannelTransform):
    """Apply Gaussian smoothing filter"""

    def __init__(self, channel_keys: Optional[List[str]] = None, sigma: float = 1.0):
        super().__init__(channel_keys)
        self.sigma = sigma

    def transform_image(self, image: np.ndarray, channel_key: str) -> np.ndarray:
        return ndimage.gaussian_filter(image, sigma=self.sigma)

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "GaussianFilterTransform",
            "channel_keys": self.channel_keys,
            "sigma": self.sigma,
        }


class SelectPlanesTransform(Transform):
    """Select specific planes from multi-plane channels"""

    def __init__(self, plane_selection: str = "middle"):
        """
        Args:
            plane_selection: 'middle', 'first', 'last', or 'all'
        """
        self.plane_selection = plane_selection

    def __call__(self, data):
        data = deepcopy(data)

        for channel_key in data.channels:
            planes = data.channels[channel_key]

            if not isinstance(planes, list) or len(planes) == 1:
                # Already single plane
                continue

            if self.plane_selection == "middle":
                selected = [planes[len(planes) // 2]]
            elif self.plane_selection == "first":
                selected = [planes[0]]
            elif self.plane_selection == "last":
                selected = [planes[-1]]
            elif self.plane_selection == "all":
                selected = planes
            else:
                raise ValueError(f"Unknown plane_selection: {self.plane_selection}")

            data.channels[channel_key] = selected

        # Also handle segmentation
        if data.segmentation is not None and isinstance(data.segmentation, list):
            planes = data.segmentation
            if self.plane_selection == "middle":
                data.segmentation = [planes[len(planes) // 2]]
            elif self.plane_selection == "first":
                data.segmentation = [planes[0]]
            elif self.plane_selection == "last":
                data.segmentation = [planes[-1]]

        if data.nuclei_segmentation is not None and isinstance(data.nuclei_segmentation, list):
            planes = data.nuclei_segmentation
            if self.plane_selection == "middle":
                data.nuclei_segmentation = [planes[len(planes) // 2]]
            elif self.plane_selection == "first":
                data.nuclei_segmentation = [planes[0]]
            elif self.plane_selection == "last":
                data.nuclei_segmentation = [planes[-1]]

        return data

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "SelectPlanesTransform",
            "plane_selection": self.plane_selection,
        }


class RollingBallTransform(ChannelTransform):
    """Apply rolling ball background subtraction"""

    def __init__(
        self,
        channel_keys: Optional[List[str]] = None,
        radius: int = 50,
        light_background: bool = False,
    ):
        """
        Args:
            channel_keys: Which channels to apply the transform to
            radius: Radius of the rolling ball (larger = smoother background)
            light_background: True if background is lighter than foreground
        """
        super().__init__(channel_keys)
        self.radius = radius

    def transform_image(self, image: np.ndarray, channel_key: str) -> np.ndarray:
        # Estimate background
        background = rolling_ball(image, radius=self.radius)

        image = image - background

        # Clip negative values
        image = np.clip(image, 0, None)

        return image

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "RollingBallTransform",
            "channel_keys": self.channel_keys,
            "radius": self.radius,
        }


class TransformPipeline(Transform):
    """Compose multiple transforms into a pipeline"""

    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms

    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "TransformPipeline",
            "transforms": [t.get_config() for t in self.transforms],
        }

    def add_transform(self, transform: Transform):
        """Add a transform to the pipeline"""
        self.transforms.append(transform)

    def __len__(self):
        return len(self.transforms)
