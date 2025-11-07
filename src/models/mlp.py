"""
MLP - Multi-Layer Perceptron for FUCCI intensity regression or general tabular regression.

This module follows the same training/validation/test logic and optimizer/scheduler
configuration style as `CNet` in `src/models/cnet.py`:
- Mean Squared Error loss with MAE/MSE logging
- R2 metric (torcheval) for validation and test
- Optimizers: adam | adamw | sgd
- Schedulers: plateau | cosine | step | None
- Learner logs current learning rate at the end of each epoch

By convention, if a batch input comes as a tuple (image_x, feature_x), this MLP ignores
image_x and uses feature_x only. If the batch is a single tensor, it is treated as features.

You can also build from a dict config using build_mlp_from_config().
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torcheval.metrics import R2Score


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.01),
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
}

_OPTIMIZERS = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}


class MLP(L.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256],
        output_dim: int = 2,
        activation: str = "relu",
        dropout: float = 0.0,
        learning_rate: float = 1e-3,
        optimizer: str = "adam",
        scheduler: Optional[str] = "plateau",
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()
        # Save hyperparameters in a way similar to CNet
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.weight_decay = weight_decay

        # Metrics and containers similar to CNet
        self.train_predictions: List[torch.Tensor] = []
        self.train_targets: List[torch.Tensor] = []
        self.val_predictions: List[torch.Tensor] = []
        self.val_targets: List[torch.Tensor] = []
        self.val_r2_score = R2Score()
        self.test_r2_score = R2Score()

        act_cls = _ACTIVATIONS.get(self.activation)
        if act_cls is None:
            raise ValueError(f"Unknown activation: {self.activation}")

        layers: List[nn.Module] = []
        prev = self.input_dim
        for h in self.hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(act_cls())
            if self.dropout and self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            prev = h
        layers.append(nn.Linear(prev, self.output_dim))
        self.net = nn.Sequential(*layers)

        # Regression criterion to match CNet's default
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure float32 for AMP compatibility
        if x.dtype != torch.float32:
            x = x.float()

        # Flatten possible extra dims, e.g., [B, 1, F] -> [B, F]
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)

    def _step(self, batch, stage: str) -> torch.Tensor:
        x, y = batch
        # Accept (image_x, feature_x) or just features; MLP uses features only
        if isinstance(x, (list, tuple)) and len(x) == 2:
            _, feature_x = x
        else:
            feature_x = x

        preds = self(feature_x)
        y = y.float()
        preds = preds.float()
        loss = self.criterion(preds, y)

        mae = F.l1_loss(preds, y)
        mse = F.mse_loss(preds, y)

        # Log metrics similar to CNet
        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=(stage == "train"),
            on_step=(stage == "train"),
            on_epoch=True,
        )
        self.log(f"{stage}_mae", mae, prog_bar=False, on_step=(stage == "train"), on_epoch=True)
        self.log(f"{stage}_mse", mse, prog_bar=False, on_step=(stage == "train"), on_epoch=True)

        if stage == "train":
            self.train_predictions.append(preds.detach().cpu())
            self.train_targets.append(y.detach().cpu())
        elif stage == "val":
            # Reset once per epoch at first batch is handled in validation_step
            self.val_predictions.append(preds.detach().cpu())
            self.val_targets.append(y.detach().cpu())
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.val_r2_score.reset()
        x, y = batch
        # Accept (image_x, feature_x) or just features
        if isinstance(x, (list, tuple)) and len(x) == 2:
            _, feature_x = x
        else:
            feature_x = x
        preds = self(feature_x)
        loss = self.criterion(preds, y.float())
        self.val_r2_score.update(preds, y)
        self._step(((feature_x), y), "val")
        return loss

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.test_r2_score.reset()
        x, y = batch
        if isinstance(x, (list, tuple)) and len(x) == 2:
            _, feature_x = x
        else:
            feature_x = x
        preds = self(feature_x)
        loss = self.criterion(preds, y.float())
        self.test_r2_score.update(preds, y)
        self._step(((feature_x), y), "test")
        return loss

    def on_validation_epoch_end(self):
        val_r2 = self.val_r2_score.compute()
        if not self.trainer.sanity_checking:
            # Mirror CNet's logger usage
            self.logger.experiment.log({"val_r2": val_r2})
        self.val_r2_score.reset()
        return val_r2

    def on_test_end(self):
        test_r2 = self.test_r2_score.compute()
        if not self.trainer.sanity_checking:
            self.logger.experiment.log({"test_r2": test_r2})
        self.test_r2_score.reset()
        return test_r2

    def predict_step(self, batch, batch_idx):
        x, _ = batch if isinstance(batch, (list, tuple)) else (batch, None)
        if isinstance(x, (list, tuple)) and len(x) == 2:
            _, feature_x = x
        else:
            feature_x = x
        preds = self(feature_x)
        return {"predictions": preds}

    def configure_optimizers(self):
        # Optimizers aligned with CNet
        opt_name = (self.optimizer_name or "adam").lower()
        if opt_name == "adam":
            optimizer = _OPTIMIZERS["adam"](self.parameters(), lr=self.learning_rate)
        elif opt_name == "adamw":
            optimizer = _OPTIMIZERS["adamw"](
                self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
        elif opt_name == "sgd":
            optimizer = _OPTIMIZERS["sgd"](
                self.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        # Schedulers aligned with CNet
        if self.scheduler_name is None:
            return optimizer

        name = (self.scheduler_name or "plateau").lower()
        if name == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, min_lr=1e-6, patience=2
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss_epoch",
                },
            }
        elif name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=100, eta_min=1e-6
            )
            return [optimizer], [scheduler]
        elif name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
            return [optimizer], [scheduler]
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler_name}")

    def on_train_epoch_end(self):
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("learning_rate", current_lr, prog_bar=False)


def build_mlp_from_config(cfg: Dict[str, Any]) -> MLP:
    """Build an MLP from a nested dict config.

    Expected structure:
            cfg = {
              "input_dim": 256,
              "hidden_dims": [512, 256, 128],
              "output_dim": 2,
              "activation": "relu",
              "dropout": 0.1,
              "learning_rate": 1e-3,
              "weight_decay": 1e-4,
              "optimizer": "adamw",
              "scheduler": "cosine",  # or "plateau" | "step" | None
            }
    """
    return MLP(
        input_dim=int(cfg["input_dim"]),
        hidden_dims=list(cfg.get("hidden_dims", [512, 256])),
        output_dim=int(cfg.get("output_dim", 2)),
        activation=str(cfg.get("activation", "relu")),
        dropout=float(cfg.get("dropout", 0.0)),
        learning_rate=float(cfg.get("learning_rate", cfg.get("lr", 1e-3))),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
        optimizer=str(cfg.get("optimizer", "adam")),
        scheduler=cfg.get("scheduler", "plateau"),
    )
