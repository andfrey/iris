"""
Base Lightning module for regression models.

Provides common training/validation/test logic that model architectures inherit from.
Models can override methods for custom behavior while inheriting the standard flow.
"""

import torch
import torch.nn as nn
import lightning as L
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
from typing import Optional, Dict, Any, Tuple
from abc import abstractmethod


class VonMisesLoss(nn.Module):
    """MLE loss for von Mises distribution."""

    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets, von_mises_kappa=None):
        """
        Compute mean squared geodesic distance.

        Args:
            predictions: Tensor [B, 1] with predicted phases
            targets: Tensor [B, 1] with true phases

        Returns:
            Mean squared geodesic distance
        """
        if von_mises_kappa is None:
            von_mises_kappa = torch.tensor(1.0, device=predictions.device)

        von_mises_kappa = von_mises_kappa.view(-1, 1)
        # If using von Mises distribution, apply circular transformation
        predictions_angle = (predictions - 0.5) * 2 * torch.pi
        targets_angle = (targets - 0.5) * 2 * torch.pi
        # Stability Trick: log(I0(k)) = k + log(i0e(k))
        log_bessel = von_mises_kappa + torch.log(torch.special.i0e(von_mises_kappa))

        # Loss = - (kappa * cos(diff) - log_bessel)
        #      = log_bessel - kappa * cos(diff)
        loss = torch.mean(
            log_bessel - von_mises_kappa * torch.cos(predictions_angle - targets_angle)
        )
        return loss


class BaseRegressionModel(L.LightningModule):
    """
    Base class for regression models with Lightning training.

    Provides:
    - Standard training/validation/test loop
    - Optimizer and scheduler configuration
    - Loss computation and logging
    - Output formatting for callbacks

    Subclasses must implement:
    - model_forward(): The actual forward pass
    - get_criterion(): The loss function

    Subclasses can optionally override:
    - _step(): Custom training step logic
    - _normalize_shapes(): Custom shape normalization
    - configure_optimizers(): Custom optimizer setup
    - Any Lightning hooks (on_train_epoch_end, etc.)

    Args:
        learning_rate: Initial learning rate
        optimizer: Optimizer type ('adam', 'adamw', 'sgd')
        scheduler: Learning rate scheduler ('plateau', 'cosine', 'step', None)
        weight_decay: Weight decay for optimizer
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        optimizer: str = "adam",
        scheduler: Optional[str] = "plateau",
        weight_decay: float = 1e-4,
        **kwargs,
    ):
        super().__init__()

        # Store training hyperparameters
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.weight_decay = weight_decay

    @abstractmethod
    def model_forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Model-specific forward pass.

        Must be implemented by subclasses to define the architecture.

        Returns:
            Predictions tensor
        """
        raise NotImplementedError("Subclasses must implement model_forward()")

    @abstractmethod
    def get_criterion(self) -> nn.Module:
        """
        Get the loss function for this model.

        Must be implemented by subclasses.

        Returns:
            Loss function module
        """
        raise NotImplementedError("Subclasses must implement get_criterion()")

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass - delegates to model_forward().

        Can be overridden for custom behavior.
        """
        return self.model_forward(*args, **kwargs)

    def _normalize_shapes(
        self, preds: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize prediction and target shapes.

        - Cast to float
        - Make 1D tensors into (B, 1) shape
        """
        y = y.float()
        preds = preds.float() if not isinstance(preds, tuple) else tuple(p.float() for p in preds)
        if y.dim() == 1:
            y = y.view(-1, 1)
        if isinstance(preds, tuple):
            if preds[0].dim() == 1 and preds[1].dim() == 1:
                preds = (preds[0].view(-1, 1), preds[1].view(-1, 1))
        elif preds.dim() == 1:
            preds = preds.view(-1, 1)
        return preds, y

    def _unpack_inputs(self, x):
        """
        Unpack batch inputs.

        Default: assumes x might be (images, features) tuple.
        Override for custom input unpacking.
        """
        if isinstance(x, (list, tuple)) and len(x) == 2:
            return x[0], x[1]
        return x, None

    def _compute_loss(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute loss and adjusted predictions.

        Default behavior for regression with optional cyclic adjustment.
        Override for custom loss computation.

        Returns:
            (loss, adjusted_predictions)
        """
        criterion = self.get_criterion()
        kappa = None
        if isinstance(criterion, VonMisesLoss):
            # For Von Mises, preds is a tuple (predictions, kappa)
            preds, kappa = preds
            loss = criterion(preds, targets, von_mises_kappa=kappa)
            abs_residuals = torch.abs(((preds - targets + 0.5) % 1.0) - 0.5)
        else:
            abs_residuals = criterion(preds, targets)
            loss = (abs_residuals**2).mean()

        # For geodesic/circular data (1D output), adjust predictions
        if preds.shape[1] == 1:
            y_hat = preds.clone() % 1.0
        else:
            y_hat = preds

        # Compute MAE and MSE for logging
        mae = abs_residuals.mean()
        mse = loss

        return loss, y_hat, mae, mse, kappa

    def _step(self, batch, stage: str) -> Dict[str, torch.Tensor]:
        """
        Shared step for training, validation, and testing.

        Override for custom step logic while maintaining callback compatibility.

        Args:
            batch: Input batch
            stage: One of "train", "val", "test"

        Returns:
            Dict with "loss", "preds", "targets" for callbacks
        """
        x, y = batch
        x = self._unpack_inputs(x)

        # Forward pass
        preds = self(*x) if isinstance(x, tuple) else self(x)

        # Normalize shapes
        preds, y = self._normalize_shapes(preds, y)

        # Compute loss
        loss, y_hat, mae, mse, kappa = self._compute_loss(preds, y)

        # Log metrics
        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=(stage == "train"),
            on_step=(stage == "train"),
            on_epoch=True,
        )
        self.log(f"{stage}_mae", mae, prog_bar=False, on_step=False, on_epoch=True)
        self.log(f"{stage}_mse", mse, prog_bar=False, on_step=False, on_epoch=True)

        # Return outputs for callbacks
        return {
            "loss": loss,
            "preds": y_hat,
            "targets": y,
            "kappa": kappa,
        }

    def training_step(self, batch, batch_idx):
        """Training step - can be overridden"""
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        """Validation step - can be overridden"""
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        """Test step - can be overridden"""
        return self._step(batch, "test")

    def predict_step(self, batch, batch_idx):
        """
        Prediction step.

        Override for custom prediction behavior.
        """
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        x = self._unpack_inputs(x)
        predictions = self(*x) if isinstance(x, tuple) else self(x)
        return {"predictions": predictions}

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.

        Override for custom optimizer/scheduler setup.
        """
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
            scheduler = CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-6)
            return [optimizer], [scheduler]
        elif self.scheduler_name.lower() == "step":
            scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
            return [optimizer], [scheduler]
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler_name}")

    def on_train_epoch_end(self):
        """
        Called at the end of training epoch.

        Override to add custom logic.
        """
        # Log learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("learning_rate", current_lr, prog_bar=False)
