import sys
from pathlib import Path

import numpy as np
from lightning.pytorch.callbacks import Callback

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.train.utils import log_regression_plots
from src.evaluation.evaluator import Evaluator, EvaluationConfig


class EvaluationCallback(Callback):
    """
    Callback that performs evaluation at the end of each epoch.

    Uses the centralized Evaluator for consistency with checkpoint evaluation.
    """

    def __init__(self, eval_config: EvaluationConfig = None):
        """
        Initialize callback.

        Args:
            eval_config: Configuration for evaluation (plots, metrics, etc.)
        """
        self.eval_config = eval_config or EvaluationConfig(plots_to_generate=[])
        self.evaluator = None  # Initialized in setup

        # Storage for predictions/targets
        self.train_outputs = []
        self.val_outputs = []
        self.test_outputs = []

    def setup(self, trainer, pl_module, stage):
        """Initialize evaluator with projector if available"""
        projector = None

        # Try to get projector from datamodule
        if hasattr(trainer, "datamodule"):
            if hasattr(trainer.datamodule, "full_dataset"):
                projector = getattr(trainer.datamodule.full_dataset, "projector", None)

        # Update config if geodesic
        if "scatter" not in self.eval_config.plots_to_generate:
            self.eval_config.plots_to_generate.append("scatter")
        if "true_residuals" not in self.eval_config.plots_to_generate:
            self.eval_config.plots_to_generate.append("true_residuals")
        if projector is not None:
            # Add geodesic-specific plots
            if "geodesic_residual" not in self.eval_config.plots_to_generate:
                self.eval_config.plots_to_generate.append("geodesic_residual")
        else:
            if "residual" not in self.eval_config.plots_to_generate:
                self.eval_config.plots_to_generate.append("residual")
        if pl_module.loss == "von_mises":
            if "von_mises_kappa_uncertainty" not in self.eval_config.plots_to_generate:
                self.eval_config.plots_to_generate.append("von_mises_kappa_uncertainty")

        self.evaluator = Evaluator(config=self.eval_config, projector=projector)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Collect training predictions"""
        if outputs is not None and "preds" in outputs and "targets" in outputs:
            self.train_outputs.append(
                {
                    "preds": outputs["preds"].detach().cpu().numpy(),
                    "targets": outputs["targets"].detach().cpu().numpy(),
                    "kappa": outputs.get("kappa").detach().cpu().numpy()
                    if outputs.get("kappa", None) is not None
                    else None,
                }
            )

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Collect validation predictions"""
        if outputs is not None and "preds" in outputs and "targets" in outputs:
            self.val_outputs.append(
                {
                    "preds": outputs["preds"].detach().cpu().numpy(),
                    "targets": outputs["targets"].detach().cpu().numpy(),
                    "kappa": outputs.get("kappa").detach().cpu().numpy()
                    if outputs.get("kappa", None) is not None
                    else None,
                }
            )

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Collect validation predictions"""
        if outputs is not None and "preds" in outputs and "targets" in outputs:
            self.test_outputs.append(
                {
                    "preds": outputs["preds"].detach().cpu().numpy(),
                    "targets": outputs["targets"].detach().cpu().numpy(),
                    "kappa": outputs.get("kappa").detach().cpu().numpy()
                    if outputs.get("kappa", None) is not None
                    else None,
                }
            )

    def on_train_epoch_end(self, trainer, pl_module):
        """Evaluate training set"""
        if not self.train_outputs:
            return
        if pl_module.loss == "von_mises":
            kappa_values = np.concatenate([x["kappa"] for x in self.train_outputs])
        else:
            kappa_values = None
        y_pred = np.concatenate([x["preds"] for x in self.train_outputs])
        y_true = np.concatenate([x["targets"] for x in self.train_outputs])

        # Use centralized evaluator
        result = self.evaluator.evaluate(y_true, y_pred, prefix="train", kappa_values=kappa_values)
        # Log to W&B
        if trainer.logger is not None:
            result.log_to_wandb(trainer.logger.experiment, prefix="train")

        # log_regression_plots(
        #     y_true,
        #     y_pred,
        #     trainer.logger.experiment,
        #     "train",
        # )
        # Clear storage
        self.train_outputs = []

    def on_validation_epoch_end(self, trainer, pl_module):
        """Evaluate validation set"""
        if not self.val_outputs or trainer.sanity_checking:
            return
        if pl_module.loss == "von_mises":
            kappa_values = np.concatenate([x["kappa"] for x in self.val_outputs])
        else:
            kappa_values = None
        y_pred = np.concatenate([x["preds"] for x in self.val_outputs])
        y_true = np.concatenate([x["targets"] for x in self.val_outputs])

        # Use centralized evaluator
        result = self.evaluator.evaluate(y_true, y_pred, prefix="val", kappa_values=kappa_values)
        # Log to W&B
        if trainer.logger is not None:
            result.log_to_wandb(trainer.logger.experiment, prefix="val")

        # log_regression_plots(
        #     y_true,
        #     y_pred,
        #     trainer.logger.experiment,
        #     "val",
        # )
        pl_module.val_predictions = []
        pl_module.val_targets = []

        # Clear storage
        self.val_outputs = []

    def on_test_epoch_end(self, trainer, pl_module):
        """Evaluate validation set"""
        if not self.test_outputs or trainer.sanity_checking:
            return
        if pl_module.loss == "von_mises":
            kappa_values = np.concatenate([x["kappa"] for x in self.test_outputs])
        else:
            kappa_values = None
        y_pred = np.concatenate([x["preds"] for x in self.test_outputs])
        y_true = np.concatenate([x["targets"] for x in self.test_outputs])

        # Use centralized evaluator
        result = self.evaluator.evaluate(y_true, y_pred, prefix="test", kappa_values=kappa_values)

        # Log to W&B
        if trainer.logger is not None:
            result.log_to_wandb(trainer.logger.experiment, prefix="test")
        # log_regression_plots(
        #     y_true,
        #     y_pred,
        #     trainer.logger.experiment,
        #     "test",
        # )
        pl_module.test_predictions = []
        pl_module.test_targets = []

        # Clear storage
        self.test_outputs = []


class DebugCallback(Callback):
    def on_sanity_check_start(self, trainer, pl_module):
        import matplotlib.pyplot as plt
        import wandb

        batch = next(iter(trainer.datamodule.train_dataloader()))
        X, y = batch
        if isinstance(X, list) or isinstance(X, tuple):
            X = X[0]
        batch_size = X.shape[0]
        channel_length = X.shape[1]
        fig, axes = plt.subplots(min(10, batch_size), min(10, channel_length), figsize=(12, 8))
        for i, ax in enumerate(axes):
            for j in range(len(ax)):
                ax[j].imshow(X[i, j].squeeze(), cmap="gray")
                ax[j].axis("off")
        trainer.logger.experiment.log({"sanity_check_images": wandb.Image(plt)})
        plt.close(fig)
