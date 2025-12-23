"""
CNet - Convolutional Neural Network for Cell Image FUCCI Intensity Regression

Inherits training logic from BaseRegressionModel.
"""

import torch
import torch.nn as nn
from typing import List, Optional

from src.models.base_model import BaseRegressionModel, VonMisesLoss


class ConvBlock(nn.Module):
    """
    Convolutional block with Conv -> BatchNorm -> ReLU -> Dropout
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_batchnorm: bool = True,
    ):
        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]

        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.ReLU())

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ABSResiduals(nn.Module):
    """
    Absolute residuals.
    """

    def __init__(self, cyclic: bool = False):
        super().__init__()
        self.cyclic = cyclic

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute residuals.
        """
        if not self.cyclic:
            return torch.abs(targets - preds)
        else:
            # preds should be in [0, 1]
            circ_pred = preds % 1.0
            abs_residuals = torch.abs((targets - circ_pred))
            # geodesic distance i.e. shortest distance on the cyclic curve with length 1
            geodesic_distance = torch.min(
                abs_residuals,
                1.0 - abs_residuals,
            )
            return geodesic_distance

    def __repr__(self):
        return f"ABSResiduals(cyclic={self.cyclic})"


class CNet(BaseRegressionModel):
    """
    CNet - Convolutional Neural Network for FUCCI intensity regression.

    Inherits training/validation/test logic from BaseRegressionModel.

    Args:
        in_channels: Number of input channels
        feature_dim: Dimension of additional features (0 if none)
        output_dim: Dimension of output (FUCCI intensities 2 and phase curve 1)
        base_filters: Number of filters in first conv layer (doubles each block)
        num_blocks: Number of convolutional blocks
        fc_hidden_dims: List[int] = [512, 256],
        dropout: float = 0.3,
        use_batchnorm: bool = True,
        learning_rate: float = 1e-3,
        optimizer: str = "adam",
        scheduler: Optional[str] = "plateau",
        weight_decay: float = 1e-4,
    """

    def __init__(
        self,
        in_channels: int = 6,
        feature_dim: int = 0,
        output_dim: int = 2,
        base_filters: int = 32,
        num_blocks: int = 4,
        fc_hidden_dims: List[int] = [512, 256],
        dropout: float = 0.3,
        use_batchnorm: bool = True,
        # Training parameters (passed to BaseRegressionModel)
        learning_rate: float = 1e-3,
        optimizer: str = "adam",
        scheduler: Optional[str] = "plateau",
        weight_decay: float = 1e-4,
        loss: str = "geodesic_squared_distance",
    ):
        # Initialize base class with training parameters
        super().__init__(
            learning_rate=learning_rate,
            optimizer=optimizer,
            scheduler=scheduler,
            weight_decay=weight_decay,
        )

        # Save all hyperparameters
        self.save_hyperparameters()

        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.output_dim = output_dim

        # Build convolutional blocks
        self.conv_blocks = nn.ModuleList()
        current_channels = in_channels

        for i in range(num_blocks):
            out_channels = base_filters * (2**i)
            self.conv_blocks.append(
                nn.Sequential(
                    ConvBlock(
                        current_channels,
                        out_channels,
                        use_batchnorm=use_batchnorm,
                    ),
                    nn.MaxPool2d(2, 2),
                )
            )
            current_channels = out_channels

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        fc_layers = []
        fc_input_dim = (
            current_channels + feature_dim
        )  # Add feature_dim if using additional features

        self.loss = loss
        # Create criterion for this model if dim 1 we predict the position on a cyclic curve
        if loss == "geodesic_squared_distance":
            self.criterion = ABSResiduals(cyclic=(output_dim == 1))
        elif loss == "von_mises":
            self.criterion = VonMisesLoss()
            output_dim += 1
        else:
            raise ValueError(f"Unsupported loss function: {loss}")

        for hidden_dim in fc_hidden_dims:
            fc_layers.extend(
                [
                    nn.Linear(fc_input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            fc_input_dim = hidden_dim

        # Output layer
        fc_layers.append(nn.Linear(fc_input_dim, output_dim))
        self.fc = nn.Sequential(*fc_layers)

    def get_criterion(self) -> nn.Module:
        """Return the loss function for this model"""
        return self.criterion

    def model_forward(
        self, image_x: torch.Tensor, feature_x: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            image_x: Input image tensor of shape (batch_size, in_channels, H, W)
            feature_x: Optional additional features tensor of shape (batch_size, feature_dim)

        Returns:
            For regression: predictions of shape (batch_size, output_dim)
        """

        if feature_x is None and isinstance(image_x, tuple):
            image_x, feature_x = image_x
        # Ensure input is float32
        if image_x.dtype != torch.float32:
            image_x = image_x.float()
        if feature_x is not None and feature_x.dtype != torch.float32:
            feature_x = feature_x.float()
        # Convolutional blocks
        for block in self.conv_blocks:
            image_x = block(image_x)

        # Global pooling
        conv_x = self.global_pool(image_x)
        conv_x = conv_x.view(conv_x.size(0), -1)

        # Concatenate tensors if features are provided
        if feature_x is not None:
            tensor1 = conv_x
            tensor2 = feature_x.view(feature_x.size(0), -1)
            x = torch.cat([tensor1, tensor2], dim=1)
        else:
            x = conv_x
        x = self.fc(x)
        if self.loss == "von_mises":
            # For von Mises, output is [predictions, kappa]
            x_pred = x[:, : self.output_dim]
            x_kappa = torch.nn.functional.softplus(x[:, -1]) + 1e-3  # Ensure kappa is positive
            x = (x_pred, x_kappa)
        return x
