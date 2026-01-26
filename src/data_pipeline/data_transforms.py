"""
Transform pipeline for data preprocessing.
Composable transforms that can be chained together.
"""

from abc import ABC, abstractmethod
import importlib
from typing import Dict, Any, List, Optional, Literal
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
        method: Literal["background_content", "background_mean"] = "background_content",
        background_padding: int = 15,
        mask: str = "cell",
    ):
        self.channel_keys = channel_keys
        self.method = method
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
                    data.channels[key] = self._remove_background(planes, masks, method=self.method)
                else:
                    data.channels[key] = self._remove_background(
                        [planes], [masks], method=self.method
                    )

        return data

    def _remove_background(
        self, images: List[np.ndarray], masks: List[np.ndarray], method: str = "background_content"
    ) -> List[np.ndarray]:
        # Subtract background
        for image, mask in zip(images, masks):
            if np.sum(mask) < 10:
                mask = masks[
                    len(masks) // 2
                ]  # Use middle plane if no mask provided has less than 10 pixels
                if mask is None:
                    raise ValueError("No segmentation mask available for background removal")
            if method == "background_content":
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
            elif method == "background_mean":
                background = image[mask == 0]
                background_mean = background.mean() if background.size > 0 else 0.0
                image[:, :] = image - background_mean
            else:
                raise ValueError(f"Unknown background removal method: {method}")
        return images

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "RemoveBackgroundTransform",
            "channel_keys": self.channel_keys,
            "background_padding": self.background_padding,
            "mask": self.mask,
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
        if data.segmentation is not None:
            planes = data.segmentation
            if self.plane_selection == "middle":
                data.segmentation = [planes[len(planes) // 2]]
            elif self.plane_selection == "first":
                data.segmentation = [planes[0]]
            elif self.plane_selection == "last":
                data.segmentation = [planes[-1]]

        if data.nuclei_segmentation is not None:
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


def create_transform_pipeline_from_config(transforms: List, transform_type: str = "image"):
    # Local import to avoid circular import

    if transform_type not in ["image", "feature"]:
        raise ValueError(f"Invalid transform pipeline type: {transform_type}")
    print(f"\nCreating {transform_type} transform pipeline")
    print(f"   - Using {transforms} transforms")

    resolved_transforms = [
        resolve(cfg["class_path"])(**cfg.get("init_args", {})) for cfg in transforms
    ]
    if transform_type == "image":
        transform_pipeline = TransformPipeline(resolved_transforms)
    else:
        from sklearn.pipeline import Pipeline

        transform_pipeline = Pipeline(
            steps=[(repr(transform), transform) for transform in resolved_transforms]
        )

    return transform_pipeline


def resolve(name: str):
    module_name, attr_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)
