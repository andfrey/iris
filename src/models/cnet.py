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
        feature_dim: int = 0,
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
        fc_input_dim = (
            current_channels + feature_dim
        )  # Add feature_dim if using additional features

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

    def forward(self, image_x, feature_x=None) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            For regression: predictions of shape (batch_size, output_dim)
            For classification: logits of shape (batch_size, output_dim)
        """

        # Ensure input is float32 for compatibility with mixed precision training
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

        # Concatenate tensors if needed
        if feature_x is not None:
            tensor1 = conv_x
            tensor2 = feature_x.view(feature_x.size(0), -1)
            x = torch.cat([tensor1, tensor2], dim=1)
        else:
            x = conv_x
        x = self.fc(x)

        return x

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch"""
        val_r2 = self.val_r2_score.compute()
        if not self.trainer.sanity_checking:
            self.logger.experiment.log({"val_r2": val_r2})
        print(f"Validation Epoch End: R2={val_r2}")
        self.val_r2_score.reset()
        super().on_validation_epoch_end()
        return val_r2

    def _normalize_shapes(
        self, preds: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Ensure prediction and target shapes are compatible.

        - Cast to float
        - If 1D, make (B,1) to match (B,1) convention for output_dim=1
        """
        y = y.float()
        preds = preds.float()
        if y.dim() == 1:
            y = y.view(-1, 1)
        if preds.dim() == 1:
            preds = preds.view(-1, 1)
        return preds, y

    def _unpack_inputs(self, x):
        if isinstance(x, (list, tuple)) and len(x) == 2:
            return x[0], x[1]
        return x, None

    def _step(self, batch, stage: str, batch_idx: int):
        """Shared step for training, validation, and testing"""
        x, y = batch
        image_x, feature_x = self._unpack_inputs(x)
        preds = self(image_x, feature_x)

        preds, y = self._normalize_shapes(preds, y)
        if preds.shape[1] == 1:
            abs_phase_loss = torch.min(
                torch.abs(y - preds) % (2 * torch.pi),
                (2 * torch.pi - (torch.abs(y - preds) % (2 * torch.pi))),
            )
            loss = (abs_phase_loss**2).mean()
            mae = abs_phase_loss.mean()
            mse = (abs_phase_loss**2).mean()
            y_hat = ((preds + torch.pi) % (2 * torch.pi)) - torch.pi
            y_hat = torch.where(y_hat - y > np.pi, y_hat - 2 * np.pi, y_hat)
            y_hat = torch.where(y - y_hat > np.pi, y_hat + 2 * np.pi, y_hat)
        else:
            loss = self.criterion(preds, y)
            mae = F.l1_loss(preds, y)
            mse = F.mse_loss(preds, y)
            y_hat = preds

        # Logging
        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=(stage == "train"),
            on_step=(stage == "train"),
            on_epoch=True,
        )
        self.log(f"{stage}_mae", mae, prog_bar=False, on_step=False, on_epoch=True)
        self.log(f"{stage}_mse", mse, prog_bar=False, on_step=False, on_epoch=True)

        if stage == "train":
            self.train_predictions.append(y_hat.detach().cpu().numpy())
            self.train_targets.append(y.detach().cpu().numpy())
        elif stage == "val":
            if batch_idx == 0:
                self.val_r2_score.reset()
            self.val_r2_score.update(y_hat, y)
            self.val_predictions.append(y_hat.detach().cpu().numpy())
            self.val_targets.append(y.detach().cpu().numpy())
            print(
                f"Validation Step Batch {batch_idx}: Loss={loss.item()}, MAE={mae.item()}, MSE={mse.item()}"
            )
        elif stage == "test":
            if batch_idx == 0:
                self.test_r2_score.reset()
            self.test_r2_score.update(y_hat, y)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train", batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val", batch_idx)

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test", batch_idx)

    def predict_step(self, batch, batch_idx):
        """Prediction step"""
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        image_x, feature_x = self._unpack_inputs(x)
        predictions = self(image_x, feature_x)
        return {"predictions": predictions}

    def on_test_end(self):
        """Called at the end of test epoch"""
        test_r2 = self.test_r2_score.compute()
        if not self.trainer.sanity_checking:
            self.logger.experiment.log({"test_r2": test_r2})
        print(f"Test Epoch End: R2={test_r2}")
        self.test_r2_score.reset()
        super().on_test_epoch_end()
        return test_r2

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
            scheduler = ReduceLROnPlateau(optimizer, min_lr=1e-6, patience=3, factor=0.7)
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
