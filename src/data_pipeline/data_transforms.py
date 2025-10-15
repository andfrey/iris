"""
Transform pipeline for data preprocessing.
Composable transforms that can be chained together.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
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
                    data.channels[key] = [
                        self.transform_image(plane) for plane in planes
                    ]
                else:
                    data.channels[key] = [self.transform_image(planes)]

        return data

    @abstractmethod
    def transform_image(self, image: np.ndarray) -> np.ndarray:
        """Transform a single 2D image"""
        pass


class CropTransform(Transform):
    """Crop image to bounding box of segmentation mask"""

    def __init__(
        self,
        padding: int = 0,
    ):
        """
        Args:
            padding: Additional pixels to include around the mask bounding box
        """
        self.padding = padding

    def __call__(self, data):
        data = deepcopy(data)

        # Get the segmentation mask to use
        mask = data.segmentation

        if mask is None:
            return data

        # Handle list of planes
        if isinstance(mask, list):
            mask = mask[0]  # Use first plane for bounding box

        # Find bounding box of non-zero regions
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not rows.any() or not cols.any():
            # Empty mask, return original data
            return data

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Add padding
        h, w = mask.shape
        rmin = max(0, rmin - self.padding)
        rmax = min(h, rmax + self.padding + 1)
        cmin = max(0, cmin - self.padding)
        cmax = min(w, cmax + self.padding + 1)

        # Crop all channels
        for key in data.channels:
            planes = data.channels[key]
            if isinstance(planes, list):
                data.channels[key] = [plane[rmin:rmax, cmin:cmax] for plane in planes]
            else:
                data.channels[key] = [planes[rmin:rmax, cmin:cmax]]

        # Crop segmentation masks
        if data.segmentation is not None:
            if isinstance(data.segmentation, list):
                data.segmentation = [
                    plane[rmin:rmax, cmin:cmax] for plane in data.segmentation
                ]
            else:
                data.segmentation = data.segmentation[rmin:rmax, cmin:cmax]

        if data.nuclei_segmentation is not None:
            if isinstance(data.nuclei_segmentation, list):
                data.nuclei_segmentation = [
                    plane[rmin:rmax, cmin:cmax] for plane in data.nuclei_segmentation
                ]
            else:
                data.nuclei_segmentation = data.nuclei_segmentation[
                    rmin:rmax, cmin:cmax
                ]

        return data

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "CropTransform",
            "padding": self.padding,
        }


class NormalizeTransform(ChannelTransform):
    """Normalize channel values"""

    def __init__(
        self,
        channel_keys: Optional[List[str]] = None,
        method: str = "minmax",  # 'minmax', 'percentile'
        clip_percentiles: tuple = (1, 99),
        target_range: tuple = (0, 1),
    ):
        super().__init__(channel_keys)
        self.method = method
        self.clip_percentiles = clip_percentiles
        self.target_range = target_range

    def transform_image(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32)

        if self.method == "minmax":
            min_val = image.min()
            max_val = image.max()
            if max_val - min_val > 0:
                normalized = (image - min_val) / (max_val - min_val)
            else:
                normalized = image

        elif self.method == "standardize":
            mean = image.mean()
            std = image.std()
            if std > 0:
                normalized = (image - mean) / std
            else:
                normalized = image - mean

        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

        # Scale to target range
        min_target, max_target = self.target_range
        normalized = normalized * (max_target - min_target) + min_target

        return normalized

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "NormalizeTransform",
            "channel_keys": self.channel_keys,
            "method": self.method,
            "clip_percentiles": self.clip_percentiles,
            "target_range": self.target_range,
        }


class GaussianFilterTransform(ChannelTransform):
    """Apply Gaussian smoothing filter"""

    def __init__(self, channel_keys: Optional[List[str]] = None, sigma: float = 1.0):
        super().__init__(channel_keys)
        self.sigma = sigma

    def transform_image(self, image: np.ndarray) -> np.ndarray:
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

        if data.nuclei_segmentation is not None and isinstance(
            data.nuclei_segmentation, list
        ):
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

    def transform_image(self, image: np.ndarray) -> np.ndarray:
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


class StackChannelsTransform(Transform):
    """Stack multiple channels into a single array"""

    def __init__(self, channel_order: List[str] = ["bf", "405"]):
        self.channel_order = channel_order

    def __call__(self, data):
        data = deepcopy(data)

        # Collect and stack channels in specified order
        stacked_planes = []
        for key in self.channel_order:
            if key in data.channels:
                planes = data.channels[key]
                # Add each plane as a channel
                for plane in planes:
                    stacked_planes.append(plane)

        if stacked_planes:
            # Stack into (C, H, W) array
            data.channels = {"stacked": np.array(stacked_planes)}

        return data

    def get_config(self) -> Dict[str, Any]:
        return {"type": "StackChannelsTransform", "channel_order": self.channel_order}


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
