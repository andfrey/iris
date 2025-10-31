"""
CNet - Convolutional Neural Network for Cell Image FUCCI Intensity Regression
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
from torcheval.metrics import R2Score
from typing import Optional, Dict, Any, List, Tuple
from src.train.utils import log_plot


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

        layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CNet(L.LightningModule):
    """
    CNet - Convolutional Neural Network for FUCCI intensity regression based on cell images.

    Args:
        in_channels: Number of input channels
        output_dim: Dimension of output (e.g., 2 for 2D regression)
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

        self.train_residuals = []
        self.train_predictions = []
        self.train_targets = []
        self.val_residuals = []
        self.val_predictions = []
        self.val_targets = []
        self.val_r2_score = R2Score()
        self.test_r2_score = R2Score()

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
        fc_input_dim = current_channels

        for hidden_dim in fc_hidden_dims:
            fc_layers.extend(
                [
                    nn.Linear(fc_input_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    # nn.Dropout(dropout),
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
        for block in self.conv_blocks:
            x = block(x)

        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc(x)

        return x

    def training_step(self, batch, batch_idx):
        """Training step"""
        x, y = batch

        # Forward pass
        predictions = self(x)
        loss = self.criterion(predictions, y)

        mae = F.l1_loss(predictions, y)
        mse = F.mse_loss(predictions, y)

        self.train_predictions.append(predictions.detach().cpu().numpy())
        self.train_targets.append(y.detach().cpu().numpy())
        # Log metrics
        self.log(f"train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"train_mae", mae, prog_bar=False, on_step=True, on_epoch=True)
        self.log(f"train_mse", mse, prog_bar=False, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        x, y = batch

        # Forward pass
        predictions = self(x)
        loss = self.criterion(predictions, y)

        mae = F.l1_loss(predictions, y)
        mse = F.mse_loss(predictions, y)

        # Log metrics
        self.log(f"val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"val_mae", mae, prog_bar=False, on_step=False, on_epoch=True)
        self.log(f"val_mse", mse, prog_bar=False, on_step=False, on_epoch=True)
        print(
            f"Validation Step Batch {batch_idx}: Loss={loss.item()}, MAE={mae.item()}, MSE={mse.item()}"
        )
        if batch_idx == 0:
            self.val_r2_score.reset()
        self.val_r2_score.update(predictions, y)

        self.val_predictions.append(predictions.detach().cpu().numpy())
        self.val_targets.append(y.detach().cpu().numpy())

        return loss

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch"""
        val_r2 = self.val_r2_score.compute()
        if not self.trainer.sanity_checking:
            self.logger.experiment.log({"val_r2": val_r2})
        print(f"Validation Epoch End: R2={val_r2}")
        self.val_r2_score.reset()
        super().on_validation_epoch_end()
        return val_r2

    def test_step(self, batch, batch_idx):
        """Test step"""
        x, y = batch

        # Forward pass
        predictions = self(x)
        loss = self.criterion(predictions, y)

        mae = F.l1_loss(predictions, y)
        mse = F.mse_loss(predictions, y)

        # Log metrics
        self.log(f"test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"test_mae", mae, prog_bar=False, on_step=False, on_epoch=True)
        self.log(f"test_mse", mse, prog_bar=False, on_step=False, on_epoch=True)

        if batch_idx == 0:
            self.test_r2_score.reset()
        self.test_r2_score.update(predictions, y)

        return loss

    def on_test_end(self):
        """Called at the end of test epoch"""
        test_r2 = self.test_r2_score.compute()
        if not self.trainer.sanity_checking:
            self.logger.experiment.log({"test_r2": test_r2})
        print(f"Test Epoch End: R2={test_r2}")
        self.test_r2_score.reset()
        super().on_test_epoch_end()
        return test_r2

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
                self.parameters(),
                lr=self.learning_rate,
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
            scheduler = ReduceLROnPlateau(optimizer, min_lr=1e-6)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
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
