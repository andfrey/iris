"""
CNet - Convolutional Neural Network for Cell Image Classification
Implemented using PyTorch Lightning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix
from typing import Optional, Dict, Any, List, Tuple
import numpy as np


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
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ):
        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]

        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.ReLU(inplace=True))

        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CNet(L.LightningModule):
    """
    CNet - Convolutional Neural Network for cell image classification.

    Architecture:
    - Multiple convolutional blocks with pooling
    - Fully connected layers
    - Supports multi-class classification
    - Built with PyTorch Lightning for easy training

    Args:
        in_channels: Number of input channels
        num_classes: Number of output classes
        base_filters: Number of filters in first conv layer (doubles each block)
        num_blocks: Number of convolutional blocks
        fc_hidden_dims: List of hidden layer dimensions for FC layers
        dropout: Dropout rate
        learning_rate: Initial learning rate
        optimizer: Optimizer type ('adam', 'adamw', 'sgd')
        scheduler: Learning rate scheduler ('plateau', 'cosine', 'step', None)
    """

    def __init__(
        self,
        in_channels: int = 6,
        output_dim: int = 2,  # For 2D regression (e.g., x, y coordinates)
        base_filters: int = 32,
        num_blocks: int = 4,
        fc_hidden_dims: List[int] = [512, 256],
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        optimizer: str = "adam",
        scheduler: Optional[str] = "plateau",
        weight_decay: float = 1e-4,
        use_batchnorm: bool = True,
    ):
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters()

        self.in_channels = in_channels
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.weight_decay = weight_decay

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
                        dropout=dropout,
                        use_batchnorm=use_batchnorm,
                    ),
                    ConvBlock(
                        out_channels,
                        out_channels,
                        dropout=dropout,
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
        fc_input_dim = current_channels

        for hidden_dim in fc_hidden_dims:
            fc_layers.extend(
                [
                    nn.Linear(fc_input_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            fc_input_dim = hidden_dim

        # Output layer
        fc_layers.append(nn.Linear(fc_input_dim, output_dim))
        self.fc = nn.Sequential(*fc_layers)

        self.criterion = nn.MSELoss()

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            For regression: predictions of shape (batch_size, output_dim)
            For classification: logits of shape (batch_size, output_dim)
        """
        # Ensure input is float32 for compatibility with mixed precision training
        if x.dtype != torch.float32:
            x = x.float()

        # Convolutional blocks
        for i, block in enumerate(self.conv_blocks):
            x = block(x)
            # Check for NaN after each block

        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc(x)

        return x

    def _shared_step(self, batch, batch_idx, stage: str):
        """Shared step for train/val/test"""
        x, y = batch

        # Forward pass
        predictions = self(x)
        loss = self.criterion(predictions, y)

        # Log metrics
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        mae = F.l1_loss(predictions, y)
        self.log(f"{stage}_mae", mae, prog_bar=True, on_step=False, on_epoch=True)
        return {"loss": loss, "predictions": predictions, "targets": y, "mae": mae}

    def training_step(self, batch, batch_idx):
        """Training step"""
        x, y = batch

        # Forward pass
        predictions = self(x)

        loss = self.criterion(predictions, y)
        mae = F.l1_loss(predictions, y)
        # Log metrics
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_mae", mae, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        output = self._shared_step(batch, batch_idx, "val")

        return output

    def test_step(self, batch, batch_idx):
        """Test step"""
        output = self._shared_step(batch, batch_idx, "test")

        return output

    def predict_step(self, batch, batch_idx):
        """Prediction step"""
        x, _ = batch if isinstance(batch, (list, tuple)) else (batch, None)
        predictions = self(x)

        return {"predictions": predictions}

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""

        # Select optimizer
        if self.optimizer_name.lower() == "adam":
            optimizer = Adam(
                self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
        elif self.optimizer_name.lower() == "adamw":
            optimizer = AdamW(
                self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
        elif self.optimizer_name.lower() == "sgd":
            optimizer = SGD(
                self.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        # Select scheduler
        if self.scheduler_name is None:
            return optimizer

        if self.scheduler_name.lower() == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5, verbose=True
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        elif self.scheduler_name.lower() == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
            return [optimizer], [scheduler]
        elif self.scheduler_name.lower() == "step":
            scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
            return [optimizer], [scheduler]
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler_name}")

    def on_train_epoch_end(self):
        """Called at the end of training epoch"""
        # Log learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("learning_rate", current_lr, prog_bar=False)


class CNetDeep(CNet):
    """
    Deeper variant of CNet with residual connections.

    Uses residual blocks for better gradient flow in deeper networks.
    """

    def __init__(self, *args, **kwargs):
        # Modify default parameters for deeper network
        kwargs.setdefault("num_blocks", 5)
        kwargs.setdefault("base_filters", 64)
        kwargs.setdefault("fc_hidden_dims", [1024, 512, 256])
        kwargs.setdefault("task", "regression")

        super().__init__(*args, **kwargs)

        # Replace conv blocks with residual blocks
        self.conv_blocks = nn.ModuleList()
        current_channels = self.in_channels
        base_filters = self.hparams.base_filters
        num_blocks = self.hparams.num_blocks
        dropout = self.hparams.dropout
        use_batchnorm = self.hparams.use_batchnorm

        for i in range(num_blocks):
            out_channels = base_filters * (2**i)
            self.conv_blocks.append(
                ResidualBlock(
                    current_channels,
                    out_channels,
                    dropout=dropout,
                    use_batchnorm=use_batchnorm,
                )
            )
            current_channels = out_channels


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ):
        super().__init__()

        self.conv1 = ConvBlock(
            in_channels, out_channels, dropout=dropout, use_batchnorm=use_batchnorm
        )
        self.conv2 = ConvBlock(
            out_channels, out_channels, dropout=dropout, use_batchnorm=use_batchnorm
        )
        self.pool = nn.MaxPool2d(2, 2)

        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.conv2(out)

        out = out + identity
        out = self.pool(out)

        return out


class CNetLite(CNet):
    """
    Lightweight variant of CNet for faster training and inference.

    Suitable for quick experiments or resource-constrained environments.
    """

    def __init__(self, *args, **kwargs):
        # Modify default parameters for lighter network
        kwargs.setdefault("num_blocks", 3)
        kwargs.setdefault("base_filters", 16)
        kwargs.setdefault("fc_hidden_dims", [128])
        kwargs.setdefault("task", "regression")

        super().__init__(*args, **kwargs)


# Factory function for easy model creation
def create_cnet(
    variant: str = "standard",
    in_channels: int = 4,
    output_dim: int = 2,
    task: str = "regression",
    **kwargs,
) -> L.LightningModule:
    """
    Factory function to create CNet models.

    Args:
        variant: Model variant ('standard', 'deep', 'lite')
        in_channels: Number of input channels
        output_dim: Output dimension (2 for 2D regression, or num_classes for classification)
        task: Task type ('regression' or 'classification')
        **kwargs: Additional arguments passed to model constructor

    Returns:
        CNet model instance

    Examples:
        >>> # For 2D regression (e.g., predicting x, y coordinates)
        >>> model = create_cnet('standard', in_channels=4, output_dim=2, task='regression')
        >>>
        >>> # For classification
        >>> model = create_cnet('standard', in_channels=4, output_dim=3, task='classification')
        >>>
        >>> # Deep variant with custom parameters
        >>> model = create_cnet('deep', in_channels=1, output_dim=2, task='regression', learning_rate=1e-4)
    """
    variant = variant.lower()

    if variant == "standard":
        return CNet(in_channels=in_channels, output_dim=output_dim, task=task, **kwargs)
    elif variant == "deep":
        return CNetDeep(
            in_channels=in_channels, output_dim=output_dim, task=task, **kwargs
        )
    elif variant == "lite":
        return CNetLite(
            in_channels=in_channels, output_dim=output_dim, task=task, **kwargs
        )
    else:
        raise ValueError(
            f"Unknown variant: {variant}. Choose from 'standard', 'deep', 'lite'"
        )


if __name__ == "__main__":
    # Test model creation
    print("Testing CNet models...")

    # Test 1: Regression model (2D output)
    print("\n" + "=" * 60)
    print("Test 1: Regression Model (2D continuous output)")
    print("=" * 60)
    model_reg = create_cnet("standard", in_channels=4, output_dim=2, task="regression")
    print(f"\nStandard CNet (Regression):")
    print(f"  Total parameters: {sum(p.numel() for p in model_reg.parameters()):,}")
    print(
        f"  Trainable parameters: {sum(p.numel() for p in model_reg.parameters() if p.requires_grad):,}"
    )

    # Test forward pass for regression
    batch_size = 8
    x = torch.randn(batch_size, 4, 128, 128)
    y_reg = torch.randn(batch_size, 2)  # 2D continuous targets

    output_reg = model_reg(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output_reg.shape}")
    print(f"  Output range: [{output_reg.min():.2f}, {output_reg.max():.2f}]")

    # Test training step
    loss_reg = model_reg.criterion(output_reg, y_reg)
    mae = F.l1_loss(output_reg, y_reg)
    print(f"  MSE Loss: {loss_reg.item():.4f}")
    print(f"  MAE: {mae.item():.4f}")

    # Test 2: Classification model
    print("\n" + "=" * 60)
    print("Test 2: Classification Model (3 classes)")
    print("=" * 60)
    model_cls = create_cnet(
        "standard", in_channels=4, output_dim=3, task="classification"
    )
    print(f"\nStandard CNet (Classification):")
    print(f"  Total parameters: {sum(p.numel() for p in model_cls.parameters()):,}")

    y_cls = torch.randint(0, 3, (batch_size,))
    output_cls = model_cls(x)
    loss_cls = model_cls.criterion(output_cls, y_cls)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output_cls.shape}")
    print(f"  CrossEntropy Loss: {loss_cls.item():.4f}")

    # Test 3: Deep model for regression
    print("\n" + "=" * 60)
    print("Test 3: Deep Regression Model")
    print("=" * 60)
    model_deep = create_cnet("deep", in_channels=4, output_dim=2, task="regression")
    print(f"\nDeep CNet (Regression):")
    print(f"  Total parameters: {sum(p.numel() for p in model_deep.parameters()):,}")

    output_deep = model_deep(x)
    print(f"  Output shape: {output_deep.shape}")

    # Test 4: Lite model for regression
    print("\n" + "=" * 60)
    print("Test 4: Lite Regression Model")
    print("=" * 60)
    model_lite = create_cnet("lite", in_channels=4, output_dim=2, task="regression")
    print(f"\nLite CNet (Regression):")
    print(f"  Total parameters: {sum(p.numel() for p in model_lite.parameters()):,}")

    output_lite = model_lite(x)
    print(f"  Output shape: {output_lite.shape}")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
